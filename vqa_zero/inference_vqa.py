import json
import os
from typing import Any, Dict, List, Tuple

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_zoo.custom_dataset import VQADataset, collate_fn_builder
from dataset_zoo.nearest_neighbor import cache_nearest_neighbor_data
from evals.answer_postprocess import postprocess_cleanup_answer
from evals.vicuna_llm import LlmEngine
from evals.vicuna_llm_evals import (extract_answers_from_predictions_vicuna,
                                    postprocess_cleanup_vicuna)
from evals.vqa_accuracy import eval_aokvqa, eval_gqa, eval_visual7w, eval_vqa
from utils.config_manager import VQAConfigManager
from utils.helpers import _cleanup, _get_all_prompts
from utils.logger import Logger
from vqa_zero.caption_generation import CaptionGenerator, PromptCapGenerarator
from vqa_zero.inference_utils import (format_last_predictions,
                                      get_output_dir_path,
                                      get_prompt_template_handler,
                                      is_vqa_output_cache_exists,
                                      load_model_and_processors,
                                      save_predictions, save_to_json,
                                      save_vqa_answers,
                                      save_vqa_answers_aggregate_chunks,
                                      set_seed)
from vqa_zero.rationale_generation import ChainOfThoughtGenerator

set_seed(42)
logger = Logger(__name__)


class VQAInference:
    def __init__(self, args):
        self.args = args

    def _get_device(self):
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _get_autocast_dtype(self):
        """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _initialize_dataloader(self, processor, tokenizer, prompt_handler):
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
            "prompt_handler": prompt_handler,
            "model_name": self.args.model_name,
            "split": self.args.split,
        }
        dataset = VQADataset(**dataset_args)
        collate_fn = collate_fn_builder(processor, tokenizer)

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _perform_vqa_inference(self) -> Dict:
        # Implementation of the VQA inference logic
        if self.args.vqa_format not in ["standard_vqa", "caption_vqa", "cot_vqa"]:
            raise NotImplementedError(
                f"Provided VQA format {self.args.vqa_format} is either not implemented yet or invalid argument provided."
            )
        device = self._get_device()
        autocast_dtype = self._get_autocast_dtype()
        self.args.autocast_dtype = autocast_dtype

        if self.args.few_shot:
            cache_nearest_neighbor_data(self.args.dataset_name, self.args.task_type == "multiple_choice")

        # get prompting handler
        prompt_handler, template_expr = get_prompt_template_handler(self.args)

        if isinstance(prompt_handler, List):
            prompt_handler, context_prompt_handler = prompt_handler
            self._handle_caption_and_cot_generation(context_prompt_handler)

        model, processor, tokenizer = load_model_and_processors(self.args.model_name, device, autocast_dtype)
        model.eval()

        batch_size = self.args.batch_size
        dataloader = self._initialize_dataloader(processor, tokenizer, prompt_handler)

        logger.info(f" Num examples: \t{len(dataloader.dataset)}")
        logger.info(f" Batch size: \t{batch_size}")
        logger.info(f" Num iterations: \t{len(dataloader.dataset) // batch_size}")

        predictions = {}
        for batch in tqdm(dataloader, desc="Batch"):
            prompt = batch["prompted_question"]
            if self.args.blind:  # zero out each images pixels, note that images are list of images
                images = [TF.to_tensor(img) * 0 for img in images]

            logger.debug(f"TEXT INPUT: {json.dumps(prompt, indent=2)}")

            if self.args.model_name in [
                "flant5xl",
                "flant5xxl",
                "opt27b",
                "opt67b",
                "vicuna13b",
                "redpajama",
                "redpajama_instruct",
            ]:
                generated_ids = model.generate(
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    max_new_tokens=self.args.max_length,
                    num_beams=self.args.num_beams,
                    length_penalty=self.args.length_penalty,
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                    do_sample=self.args.do_sample,
                    temperature=self.args.temperature,
                    num_return_sequences=self.args.num_return_sequences,
                )
                generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                if processor is not None and processor.padding_side == "left":
                    generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
                    generated_outputs = [processor.post_process_generation(gt) for gt in generated_outputs]

            elif self.args.model_name == "kosmos2":
                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"].to("cuda"),
                    input_ids=batch["input_ids"][:, :-1].to("cuda"),
                    attention_mask=batch["attention_mask"][:, :-1].to("cuda"),
                    img_features=None,
                    img_attn_mask=batch["img_attn_mask"][:, :-1].to("cuda"),
                    max_new_tokens=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    num_beams=self.args.num_beams,
                    do_sample=self.args.do_sample,
                    temperature=self.args.temperature,
                    num_return_sequences=self.args.num_return_sequences,
                )
                generated_ids = generated_ids[:, batch["input_ids"].shape[1] - 1 :]
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                generated_outputs, entities_list = zip(
                    *[processor.post_process_generation(gt) for gt in generated_texts]
                )

            elif "open_flamingo" in self.args.model_name:
                # TODO: torch.bfloat16 `dtype` is not used here
                batch_images = batch["image_tensors"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    generated_ids = model.generate(
                        vision_x=batch_images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        min_new_tokens=0,
                        max_new_tokens=self.args.max_length,
                        num_beams=self.args.num_beams,
                        length_penalty=self.args.length_penalty,
                        do_sample=self.args.do_sample,
                        temperature=self.args.temperature,
                        num_return_sequences=self.args.num_return_sequences,
                    )
                generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
                generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            elif self.args.model_name == "llava":
                from LLaVA.llava.conversation import (SeparatorStyle,
                                                      conv_templates)
                from LLaVA.llava.mm_utils import KeywordsStoppingCriteria

                conv = conv_templates[processor.conv_mode].copy()
                stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
                keywords = [stop_str]
                stopping_criteria = (
                    [KeywordsStoppingCriteria(keywords, processor.tokenizer, input_ids)]
                    if conv.version == "v0"
                    else None
                )
                input_ids = batch["input_ids"]
                image_tensor = batch["image_tensors"]
                input_ids = input_ids.cuda()

                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.half().cuda(),
                    num_beams=self.args.num_beams,
                    max_new_tokens=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    use_cache=True,
                    stopping_criteria=stopping_criteria,
                    do_sample=self.args.do_sample,
                    temperature=self.args.temperature,
                    num_return_sequences=self.args.num_return_sequences,
                )
                generated_outputs = processor.tokenizer.batch_decode(
                    output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
                )
                generated_outputs = [out.strip() for out in generated_outputs]
                generated_outputs = [
                    out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs
                ]

            else:
                inputs = {
                    "input_ids": batch["input_ids"].to(model.device),
                    "attention_mask": batch["attention_mask"].to(model.device),
                    "pixel_values": batch["pixel_values"].to(model.device, autocast_dtype),
                }
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.args.max_length,
                    num_beams=self.args.num_beams,
                    length_penalty=self.args.length_penalty,
                    no_repeat_ngram_size=self.args.no_repeat_ngram_size,
                    do_sample=self.args.do_sample,
                    temperature=self.args.temperature,
                    num_return_sequences=self.args.num_return_sequences,
                )
                generated_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            keys = ["question", "prompted_question", "question_id", "answer"]

            for i in range(0, len(generated_outputs), self.args.num_return_sequences):
                j = i // self.args.num_return_sequences  # integer division to get the group index (self-consistency)
                example = {key: batch[key][j] for key in keys}
                example["generated_output"] = (
                    generated_outputs[i]
                    if self.args.num_return_sequences == 1
                    else generated_outputs[i : i + self.args.num_return_sequences]
                )
                predictions[example["question_id"]] = example

            formatted_message = format_last_predictions(
                batch["question"], generated_outputs, self.args.num_return_sequences
            )
            logger.debug(formatted_message)

        return predictions

    def _evaluate_predictions(self, predictions: Dict, **kwargs) -> None:
        # Implementation of the evaluation logic
        logger.info("Running evaluation...")

        predictions = postprocess_cleanup_answer(self.args, predictions, logger)
        output_dir = get_output_dir_path(self.args, **kwargs)

        save_to_json(os.path.join(output_dir, "predictions.json"), predictions)  # for debugging
        save_to_json(os.path.join(output_dir, "configs.json"), vars(self.args))
        self._save_and_evaluate_predictions(output_dir, predictions)

    def _save_and_evaluate_predictions(self, output_dir, predictions: Dict, vicuna_ans_parser=False) -> None:
        if self.args.chunk_id is not None:  # chunked evaluation for vqa_v2
            save_vqa_answers_aggregate_chunks(output_dir, self.args.dataset_name, vicuna_ans_parser=vicuna_ans_parser)

            if os.path.join(output_dir, "annotations+vqa_answers.json"):
                eval_vqa(self.args, output_dir, vicuna_ans_parser)
            return

        multiple_choice = self.args.task_type == "multiple_choice"
        dataset_eval_mapping = {
            "okvqa": (save_vqa_answers, eval_vqa),
            "vqa_v2": (save_vqa_answers, eval_vqa),  # eval_vqa might not work for vqa_v2 because of chunking
            "gqa": (None, eval_gqa),
            "visual7w": (
                None,
                lambda args, dir, parser: eval_visual7w(
                    args, dir, predictions, multiple_choice=multiple_choice, vicuna_ans_parser=parser
                ),
            ),
            "aokvqa": (
                None,
                lambda args, dir, parser: eval_aokvqa(
                    args, dir, predictions, multiple_choice=multiple_choice, vicuna_ans_parser=parser
                ),
            ),
        }

        save_func, eval_func = dataset_eval_mapping.get(self.args.dataset_name, (None, None))

        if save_func:
            save_func(output_dir, self.args.dataset_name, predictions, self.args.vicuna_ans_parser)

        if eval_func:
            eval_func(self.args, output_dir, vicuna_ans_parser)

    def _handle_caption_and_cot_generation(self, prompt_handler) -> None:
        # Implementation of the caption and chain-of-thought generation logic
        if self.args.gen_model_name is None:
            self.args.gen_model_name = self.args.model_name

        def get_caption_generator(args):
            if args.gen_model_name == "promptcap":
                return PromptCapGenerarator(args, device="cuda")
            else:
                return CaptionGenerator(args, device="cuda")

        train_split = "train_bal" if self.args.dataset_name == "gqa" else "train"
        if self.args.vqa_format == "caption_vqa":
            caption_gen = get_caption_generator(self.args)
            if self.args.few_shot:
                caption_gen.generate_caption(prompt_handler, train_split)
            caption_gen.generate_caption(prompt_handler, self.args.split)

        elif self.args.vqa_format == "cot_vqa":
            cot_gen = ChainOfThoughtGenerator(self.args, device="cuda")
            if self.args.few_shot:
                cot_gen.generate_chain_of_thought_rationale(prompt_handler, train_split)

            if "iterative" in self.args.prompt_name or "mixer" in self.args.prompt_name:
                cot_gen.generate_chain_of_thought_rationale(prompt_handler, self.args.split)

    def run(self) -> None:
        config_manager = VQAConfigManager(self.args)
        all_prompts = [self.args.prompt_name]
        if not self.args.prompt_name:
            all_prompts = _get_all_prompts(self.args)

        for prompt_name in all_prompts:
            self.args.prompt_name = prompt_name

            if is_vqa_output_cache_exists(self.args):
                continue

            self.args = config_manager.apply_updates_to_args()
            predictions = self._perform_vqa_inference()
            self._evaluate_predictions(predictions)
            _cleanup()


class VicunaInference(VQAInference):
    def __init__(self, args):
        super().__init__(args)
        self.language_model_name = "Open-Orca/Mistral-7B-OpenOrca"
        self.answer_extractor = LlmEngine(language_model_name=self.language_model_name, device="cuda")

    def _evaluate_predictions_vicuna(self) -> None:
        # Implementation of the Vicuna evaluation logic
        logger.info("Running evaluation (vicunallm)...")

        output_dir = get_output_dir_path(self.args)
        fpath = os.path.join(output_dir, "predictions.json")
        with open(fpath, "r") as f:
            predictions = json.load(f)

        batch_size = 16 if "rationale" in self.args.prompt_name else 32
        predictions = extract_answers_from_predictions_vicuna(predictions, self.answer_extractor, batch_size=batch_size)
        predictions = postprocess_cleanup_vicuna(self.args.dataset_name, predictions)

        save_to_json(fpath, predictions)  # for debugging
        self._save_and_evaluate_predictions(output_dir, predictions, vicuna_ans_parser=True)

    def _check_and_reinitialize_model(self) -> None:
        """
        Check and reinitialize the model based on the prompt name.
        """
        model_name_mapping = {True: "lmsys/vicuna-13b-v1.5", False: "Open-Orca/Mistral-7B-OpenOrca"}

        is_rationale_in_prompt = "rationale" in self.args.prompt_name
        desired_model_name = model_name_mapping[is_rationale_in_prompt]

        if self.answer_extractor.language_model_name != desired_model_name:
            self.answer_extractor.language_model_name = desired_model_name
            self.answer_extractor.load_pretrained_model_tokenizer()

    def run(self) -> None:
        super().run()

        self.answer_extractor.load_pretrained_model_tokenizer()

        all_prompts = [self.args.prompt_name]
        if not self.args.prompt_name:
            all_prompts = _get_all_prompts(self.args)

        for prompt_name in all_prompts:
            self.args.prompt_name = prompt_name
            # self._check_and_reinitialize_model()  # hack to reinitialize vicuna llm for rationale and non-rationale prompts
            format_type = "chain_of_thought_samples" if "rationale" in prompt_name else "samples"
            self.answer_extractor._load_in_context_examples(
                dataset_name=self.args.dataset_name, format_type=format_type
            )
            self._evaluate_predictions_vicuna()


def main(args: Any) -> None:
    if args.vicuna_ans_parser:
        inference = VicunaInference(args)
    else:
        inference = VQAInference(args)

    inference.run()
