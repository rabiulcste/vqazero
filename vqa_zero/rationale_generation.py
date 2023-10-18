import json
import os
import re
import time
from typing import List, Union

import torch
from thefuzz import fuzz
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from dataset_zoo.custom_dataset import VQADataset, collate_fn
from evals.answer_postprocess import extract_answer_from_cot
from utils.config import OUTPUT_DIR
from utils.globals import MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.logger import Logger

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

    def generate_chain_of_thought_rationale(self, prompt_handler: PromptingHandler, split):
        if os.path.exists(self.get_file_path(split)):
            file_mod_time = os.path.getmtime(self.get_file_path(split))
            current_time = time.time()
            two_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < two_days_in_seconds:
                logger.info(f"Rationale data already exists. You can load it from cache {self.get_file_path()}")
                return

        model_name = MODEL_CLS_INFO["hfformer"][self.args.gen_model_name]["name"]
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        batch_size = 32
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
            "prompt_handler": prompt_handler,
            "model_name": self.args.gen_model_name,
            "split": split,
        }
        dataset = VQADataset(**dataset_args)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,  # set to True to enable faster data transfer between CPU and GPU
        )

        logger.info(
            f" Generating rationale data for {self.args.dataset_name} dataset, {split} split."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        logger.info(f" Num examples: \t{len(dataset)}")
        logger.info(f" Batch size: \t{batch_size}")
        logger.info(f" Num iterations: \t{len(dataset) // batch_size}")

        success = 0
        output = {}
        for batch in tqdm((dataloader), desc="Generating rationales"):
            images = batch["image"]
            questions = batch["question"]
            prompt = batch["prompted_question"]
            inputs = self.processor(images=images, text=prompt, padding=True, return_tensors="pt").to(
                self.device, torch.bfloat16
            )

            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                length_penalty=1.4,
                no_repeat_ngram_size=3,
            )
            generated_rationales = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
            assert len(generated_rationales) == len(questions)

            for idx, (question_id, question, rationale) in enumerate(
                zip(batch["question_id"], questions, generated_rationales)
            ):
                prediction = extract_answer_from_cot(rationale)
                answer = batch["answer"][idx]

                fuzz_score = fuzz.ratio(answer, prediction)
                if self.args.vqa_format == "cot_qa" and fuzz_score > 80:
                    output[question_id] = rationale
                    logger.info(
                        f"Question = {question}, Rationale: {rationale}, Prediction = {prediction}, Answer = {answer}"
                    )
                    success += 1
                else:
                    output[question_id] = rationale
                    success += 1

        logger.info(f"Generated rationales for {success} questions out of {len(dataset)} questions.")

        self.save(output, split)

    def generate_blip(self, samples):
        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.args.model_name == "blip_vqa":
                return self.model.generate(samples=samples)
            else:
                return self.model.generate(
                    samples=samples, max_length=100, length_penalty=1.4, num_beams=5, no_repeat_ngram_size=3
                )

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
        fname = self.get_file_path()
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

    def remove_answer_from_rationale(self, rationale, stragety="iterative"):
        rationale = rationale.replace("To answer the question, consider the following:", "")
        rationale = rationale.replace("To answer the above question, the relevant sentence is:", "")
        rationale = rationale.replace("To answer this question, we should know that", "")
        extracted_answer = extract_answer_from_cot(rationale)
        # logger.info(f"Rationale: {rationale} | Extracted answer: {extracted_answer}")
        if extracted_answer:
            if stragety == "iterative":
                match = re.search(r"\. (?=[A-Z])", rationale)
                if match:
                    rationale = rationale[: match.end() - 1]
            else:
                # Find the index of the extracted answer and remove the sentence containing it
                answer_start_index = rationale.rfind(extracted_answer)
                last_period_index = rationale[:answer_start_index].rfind(".")
                rationale = rationale[: last_period_index + 1].strip()
        # logger.info(f"Rationale after removing answer: {rationale}")
        return rationale

    def load_by_ids(self, ids: Union[str, List[str]]):
        if isinstance(ids, str):
            prompted_rationales_batch = self.data[str(ids)]  # handling winoground case
        else:
            prompted_rationales_batch = [self.data[str(idx)] for idx in ids]
        return prompted_rationales_batch

    def save(self, output_rationales, split):
        fname = self.get_file_path(split)
        with open(fname, "w") as f:
            json.dump(output_rationales, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved generated rationales to {fname}")

    def get_file_path(self, split) -> str:
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
            split,
        )
        os.makedirs(dir_path, exist_ok=True)

        file_path = os.path.join(dir_path, f"output.json")
        return file_path

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
