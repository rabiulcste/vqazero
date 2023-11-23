import copy
import json
import os
from typing import List, Union

import torch
from thefuzz import fuzz
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_zoo.custom_dataset import VQADataset, collate_fn_builder
from evals.answer_postprocess import extract_answer_from_cot
from utils.config import OUTPUT_DIR
from utils.globals import MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.logger import Logger
from vqa_zero.inference_utils import load_model_and_processors

logger = Logger(__name__)


N = 5


class ChainOfThoughtGenerator:
    """
    This class class generates chain-of-thought rationales for VQA questions.
    """

    def __init__(self, args, device="cuda"):
        self.args = args
        self.data = None  # rationale data
        self.device = device
        self.gpu_count = torch.cuda.device_count()

    def _initialize_dataloader(self, split_name: str, prompt_handler: PromptingHandler, collate_fn):
        config = copy.copy(self.args)
        if "train" in split_name:  # hack to avoid chunking for train split for few-shot inference (cvpr submission)
            config.chunk_id = None

        batch_size = 3 if self.args.gen_model_name == "llava" else 20 * self.gpu_count  # hardcoded for now
        dataset_args = {
            "config": config,
            "dataset_name": self.args.dataset_name,
            "prompt_handler": prompt_handler,
            "split_name": split_name,
            "cache_init": False,
        }
        dataset = VQADataset(**dataset_args)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def generate_chain_of_thought_rationale(self, prompt_handler: PromptingHandler, split_name: str):
        rationale_file_path = self.get_output_file_name(split_name)
        if os.path.exists(rationale_file_path):
            logger.info(f"Rationale data already exists. You can load it from cache {rationale_file_path}")
            return

        self.model, self.processor, self.tokenizer = load_model_and_processors(
            self.args.gen_model_name, self.args.device, self.args.autocast_dtype
        )
        collate_fn = collate_fn_builder(self.processor, self.tokenizer)
        dataloader = self._initialize_dataloader(split_name, prompt_handler, collate_fn)

        logger.info(
            f" Generating rationale data for {self.args.dataset_name} dataset, {split_name} split."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        logger.info(f" Num examples: \t{len(dataloader.dataset)}")
        logger.info(f" Batch size: \t{dataloader.batch_size}")
        logger.info(f" Num iterations: \t{len(dataloader.dataset) // dataloader.batch_size}")

        success = 0
        results, cached_ids = self.load_data_from_cache(split_name)

        for batch in tqdm((dataloader), desc="Generating rationales"):
            questions = batch["question"]
            question_ids = batch["question_id"]

            if all([qid in cached_ids for qid in question_ids]):
                continue

            generated_rationales = self._generate(batch)
            assert len(generated_rationales) == len(questions), f"{len(generated_rationales)} != {len(questions)}"

            for idx, (question_id, question, rationale) in enumerate(
                zip(batch["question_id"], questions, generated_rationales)
            ):
                prediction = extract_answer_from_cot(rationale)
                answer = batch["answer"][idx]

                fuzz_score = fuzz.ratio(answer, prediction)
                if self.args.vqa_format == "cot_vqa" and fuzz_score > 80 and split_name == "train":
                    results[question_id] = rationale
                    success += 1
                else:
                    results[question_id] = rationale
                    success += 1
                logger.info(
                    f"Question = {question}, Rationale: {rationale}, Prediction = {prediction}, Answer = {answer}"
                )
            logger.info(f"Generated rationales for {success} questions out of {len(dataloader.dataset)} questions.")

            if len(results) % 10 == 0:
                cache_fname = self.get_cache_file_path(split_name)
                self.save_to_json(results, cache_fname)

        self.save_to_json(results, rationale_file_path)

    def _generate(self, batch):
        print(self.args.gen_model_name)
        if self.args.gen_model_name.startswith("blip"):
            inputs = {
                "input_ids": batch["input_ids"].to(self.model.device),
                "attention_mask": batch["attention_mask"].to(self.model.device),
                "pixel_values": batch["pixel_values"].to(self.model.device, self.args.autocast_dtype),
            }
            generated_rationales = self.generate_blip2(inputs)

        elif self.args.gen_model_name == "llava":
            generated_rationales = self.generate_llava(batch)

        return generated_rationales

    def generate_blip(self, inputs):
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=3,
            length_penalty=1,
            no_repeat_ngram_size=3,
        )
        generated_rationales = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_rationales

    def generate_hfformer(self, samples, max_new_tokens, num_beams, length_penalty, no_repeat_ngram_size):
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated_ids = self.model.generate(
                **samples,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
            output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return output

    def generate_llava(self, batch, max_length=128, do_sample=True, num_return_sequences=1):
        from LLaVA.llava.conversation import SeparatorStyle, conv_templates
        from LLaVA.llava.mm_utils import KeywordsStoppingCriteria

        conv = conv_templates[self.processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, self.processor.tokenizer, input_ids)] if conv.version == "v0" else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        input_ids = input_ids.cuda()

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            use_cache=True,
            do_sample=do_sample,
            max_new_tokens=max_length,
            stopping_criteria=stopping_criteria,
            num_return_sequences=num_return_sequences,
        )
        generated_outputs = self.processor.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )
        return generated_outputs

    def generate_iterative(
        self,
        image_tensors,
        questions: List[str],
        max_new_tokens=50,
        num_beams=5,
        length_penalty=0.6,
        no_repeat_ngram_size=3,
    ):
        if self.args.model_name.startswith("blip"):
            generated_rationales = self.generate_iterative_blip(image_tensors, questions)
        else:
            generated_rationales = self.generate_iterative_git(
                image_tensors, questions, max_new_tokens, num_beams, length_penalty, no_repeat_ngram_size
            )

        return generated_rationales

    def filter_generated_ratioanles(self, generated_rationales, questions):
        raise NotImplementedError(f"filter_generated_ratioanles not implemented for {self.args.model_name}")

    def load(self):
        fname = self.get_output_file_name()
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"File {fname} does not exist. Please generate the prompted rationales first.\n"
                f"python rationale_generation.py --dataset_name {self.args.dataset_name} "
                f"--model_name {self.args.model_name} --vqa_format cot_qa "
                f"--prompt_name {self.args.prompt_name}"
            )

        with open(fname, "r") as f:
            prompted_rationales = json.load(f)
        logger.info(f"Loaded prompted rationales from {fname}")

        if "mixer" in self.args.prompt_name:
            answer_removal_stragety = "mixer"
        else:
            answer_removal_stragety = "iterative"

        for question_id in prompted_rationales:
            prompted_rationales[question_id] = self.remove_answer_from_rationale(
                prompted_rationales[question_id], answer_removal_stragety
            )
        self.data = prompted_rationales

    def load_by_ids(self, ids: Union[str, List[str]]):
        if isinstance(ids, str):
            prompted_rationales_batch = self.data[str(ids)]  # handling winoground case
        else:
            prompted_rationales_batch = [self.data[str(idx)] for idx in ids]
        return prompted_rationales_batch

    def load_data_from_cache(self, data_split: str):
        cached_data = {}
        cache_file_path = self.get_cache_file_path(data_split)
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                cached_data = json.load(file)
            logger.info(f"Loaded cached data from {cache_file_path}")
        cached_ids = set(cached_data.keys())
        return cached_data, cached_ids

    def save_to_json(self, output_rationales, fname: str):
        with open(fname, "w") as f:
            json.dump(output_rationales, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved generated rationales to {fname}")

    def get_output_dir_path(self, split_name: str) -> str:
        required_args = [self.args.dataset_name, self.args.gen_model_name, self.args.vqa_format, self.args.prompt_name]
        if any(val is None for val in required_args):
            raise ValueError(
                f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
                f"Provided: {required_args}"
            )
        if "," in self.args.prompt_name:
            subdir_prefix = self.args.prompt_name.split(",")[1]
        else:
            subdir_prefix = self.args.prompt_name

        dir_path = os.path.join(
            OUTPUT_DIR,
            "cache",
            "generated_rationale_dumps",
            self.args.dataset_name,
            self.args.gen_model_name,
            self.args.vqa_format,
            subdir_prefix,
            split_name,
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def get_output_file_name(self, split_name: str) -> str:
        dir_path = self.get_output_dir_path(split_name)
        output_file_path = os.path.join(dir_path, f"output.json")
        return output_file_path

    def get_cache_file_path(self, split_name: str) -> str:
        dir_path = self.get_output_dir_path(split_name)
        cache_file_name = os.path.join(dir_path, f"cached_output.json")
        return cache_file_name

    @staticmethod
    def get_prompt_str_for_prefix_rationale(prompt_handler: PromptingHandler, questions: List[str]) -> List[str]:
        if "promptcap" in prompt_handler.prompt_name:
            prompt_txt = []
            for q in questions:
                prompt_txt.append(prompt_handler.prompt.apply({"question": q})[0])
        elif prompt_handler.prompt_name.startswith("prefix_"):
            prompt_txt = prompt_handler.prompt.apply({})[0]
            logger.debug(f"PROMPT FOR RATIONALE GENERATION: {prompt_txt}")
        else:
            prompt_txt = ""

        if isinstance(prompt_txt, str):
            prompt_txt = [prompt_txt] * len(questions)
        return prompt_txt
