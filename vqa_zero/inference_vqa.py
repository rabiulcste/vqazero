import json
import os
import re
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
                                      is_vqa_output_vicuna_cache_exists,
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

    @staticmethod
    def get_device():
        return "cuda" if torch.cuda.is_available() else "cpu"

    @staticmethod
    def get_autocast_dtype():
        """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                return torch.bfloat16
        return torch.float16

    def _initialize_dataloader(self, prompt_handler, collate_fn):
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
            "prompt_handler": prompt_handler,
            "model_name": self.args.model_name,
            "split_name": self.args.split_name,
        }
        dataset = VQADataset(**dataset_args)

        return DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def _perform_vqa_inference(self) -> Dict:
        if self.args.vqa_format not in ["standard_vqa", "caption_vqa", "cot_vqa"]:
            raise NotImplementedError(
                f"Provided VQA format {self.args.vqa_format} is either not implemented yet or invalid argument provided."
            )
        device = self.get_device()
        autocast_dtype = self.get_autocast_dtype()
        self.args.autocast_dtype = autocast_dtype

        if self.args.few_shot:
            cache_nearest_neighbor_data(
                self.args.dataset_name, self.args.task_type == "multiple_choice", self.args.nearest_neighbor_threshold
            )

        # get prompting handler
        prompt_handler, template_expr = get_prompt_template_handler(self.args)

        if isinstance(prompt_handler, List):
            prompt_handler, context_prompt_handler = prompt_handler
            self._handle_caption_and_cot_generation(context_prompt_handler)

        model, processor, tokenizer = load_model_and_processors(self.args.model_name, device, autocast_dtype)
        model.eval()

        batch_size = self.args.batch_size
        collate_fn = collate_fn_builder(processor, tokenizer)
        dataloader = self._initialize_dataloader(prompt_handler, collate_fn)

        logger.info(f" Num examples: \t{len(dataloader.dataset)}")
        logger.info(f" Batch size: \t{batch_size}")
        logger.info(f" Num iterations: \t{len(dataloader.dataset) // batch_size}")

        predictions = {}
        for batch in tqdm(dataloader, desc="Batch"):
            prompt = batch["prompted_question"]

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
                if processor is not None and processor.tokenizer.padding_side == "left":
                    generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
                elif tokenizer is not None and tokenizer.padding_side == "left":
                    generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]

                generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            elif self.args.model_name == "kosmos2":
                generated_ids = model.generate(
                    pixel_values=batch["pixel_values"].to("cuda"),
                    input_ids=batch["input_ids"].to("cuda"),
                    attention_mask=batch["attention_mask"].to("cuda"),
                    image_embeds=None,
                    image_embeds_position_mask=batch["image_embeds_position_mask"].to("cuda"),
                    use_cache=True,
                    max_new_tokens=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    num_beams=self.args.num_beams,
                    do_sample=self.args.do_sample,
                    temperature=self.args.temperature,
                    num_return_sequences=self.args.num_return_sequences,
                )
                generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
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

            elif "minigpt4" in self.args.model_name:
                from MiniGPT4.minigpt4.common.eval_utils import prepare_texts
                from MiniGPT4.minigpt4.conversation.conversation import \
                    CONV_VISION_minigptv2

                conv_temp = CONV_VISION_minigptv2.copy()

                questions = batch["prompted_question"]
                images = batch["image_tensors"]

                texts = prepare_texts(questions, conv_temp)  # warp the texts with conversation template
                generated_outputs = model.generate(
                    images, texts, max_new_tokens=self.args.max_length, do_sample=self.args.do_sample
                )

            elif "llava" in self.args.model_name:
                input_ids = batch["input_ids"]
                image_tensor = batch["image_tensors"]
                input_ids = input_ids.to("cuda", non_blocking=True)

                with torch.inference_mode():
                    generated_ids = model.generate(
                        input_ids,
                        images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                        num_beams=self.args.num_beams,
                        max_new_tokens=self.args.max_length,
                        length_penalty=self.args.length_penalty,
                        use_cache=True,
                        do_sample=self.args.do_sample,
                        temperature=self.args.temperature,
                        num_return_sequences=self.args.num_return_sequences,
                    )
                generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
                generated_outputs = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

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

            generated_outputs = [out.strip() for out in generated_outputs]
            for i in range(0, len(generated_outputs), self.args.num_return_sequences):
                j = i // self.args.num_return_sequences  # integer division to get the group index (self-consistency)
                example = {key: batch[key][j] for key in keys}
                example["generated_output"] = (
                    generated_outputs[i]
                    if self.args.num_return_sequences == 1
                    else generated_outputs[i : i + self.args.num_return_sequences]
                )
                predictions[example["question_id"]] = example
                logger.debug(f"GENERATED OUTPUT: {json.dumps(example, indent=2)}")

            formatted_message = format_last_predictions(
                batch["question"], generated_outputs, self.args.num_return_sequences
            )
            logger.info(formatted_message)

        return predictions

    def _evaluate_predictions(self, predictions: Dict, **kwargs) -> None:
        logger.info("Running evaluation...")

        predictions = postprocess_cleanup_answer(self.args, predictions, logger)
        output_dir = get_output_dir_path(self.args, **kwargs)

        prediction_file_name = os.path.join(output_dir, "predictions.json")
        run_config_file_name = os.path.join(output_dir, "configs.json")
        save_to_json(prediction_file_name, predictions)  # for debugging
        save_to_json(run_config_file_name, vars(self.args))
        self._save_and_evaluate_predictions(output_dir, predictions)

    def _save_and_evaluate_predictions(self, output_dir, predictions: Dict, vicuna_ans_parser=False) -> None:
        if self.args.chunk_id is not None:  # chunked evaluation for vqa_v2
            save_vqa_answers_aggregate_chunks(output_dir, self.args.dataset_name, vicuna_ans_parser=vicuna_ans_parser)
            aggregated_output_dir = output_dir.replace(f"chunked/chunk{self.args.chunk_id}", "")
            final_answer_file_name = os.path.join(aggregated_output_dir, "annotations+vqa_answers.json")
            if os.path.exists(final_answer_file_name):
                eval_vqa(self.args, aggregated_output_dir, vicuna_ans_parser)
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

    def _get_train_split_name(self):
        if self.args.dataset_name == "gqa":
            train_split = "train_bal"
        elif self.args.dataset_name == "vqa_v2":
            train_split = "train_30k"
        else:
            train_split = "train"

        return train_split

    def _handle_caption_and_cot_generation(self, prompt_handler) -> None:
        if self.args.gen_model_name is None:
            self.args.gen_model_name = self.args.model_name

        train_split_name = self._get_train_split_name()
        eval_split_name = self.args.split_name

        def get_caption_generator(args):
            if args.gen_model_name == "promptcap":
                return PromptCapGenerarator(args, device="cuda")
            else:
                return CaptionGenerator(args, device="cuda")

        if self.args.vqa_format == "caption_vqa":
            caption_gen = get_caption_generator(self.args)
            if self.args.few_shot:
                caption_gen.generate_caption(prompt_handler, train_split_name)
            caption_gen.generate_caption(prompt_handler, eval_split_name)

        elif self.args.vqa_format == "cot_vqa":
            cot_gen = ChainOfThoughtGenerator(self.args, device="cuda")
            if self.args.few_shot:
                cot_gen.generate_chain_of_thought_rationale(prompt_handler, train_split_name)

            if "rationale" in self.args.prompt_name and (
                "iterative" in self.args.prompt_name or "mixer" in self.args.prompt_name
            ):
                cot_gen.generate_chain_of_thought_rationale(prompt_handler, eval_split_name)

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
        self.language_model_name = "HuggingFaceH4/zephyr-7b-beta"
        self.answer_extractor = LlmEngine(language_model_name=self.language_model_name, device="cuda")

    def _evaluate_predictions_vicuna(self) -> None:
        logger.info("Running evaluation (vicunallm)...")

        output_dir = get_output_dir_path(self.args)
        prediction_file_name = os.path.join(output_dir, "predictions.json")
        with open(prediction_file_name, "r") as f:
            predictions = json.load(f)

        batch_size = 16 if "rationale" in self.args.prompt_name else 32
        batch_size = torch.cuda.device_count() * batch_size
        predictions = extract_answers_from_predictions_vicuna(predictions, self.answer_extractor, batch_size=batch_size)
        predictions = postprocess_cleanup_vicuna(self.args.dataset_name, predictions)

        save_to_json(prediction_file_name, predictions)  # for debugging
        self._save_and_evaluate_predictions(output_dir, predictions, vicuna_ans_parser=True)

    def run(self) -> None:
        super().run()

        self.answer_extractor.load_pretrained_model_tokenizer()

        all_prompts = [self.args.prompt_name]
        if not self.args.prompt_name:
            all_prompts = _get_all_prompts(self.args)

        for prompt_name in all_prompts:
            self.args.prompt_name = prompt_name
            if is_vqa_output_vicuna_cache_exists(self.args):
                continue
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
