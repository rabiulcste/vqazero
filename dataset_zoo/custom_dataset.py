import json
import os
import random
from collections import defaultdict
from string import punctuation
from typing import Dict, List, Union

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_zoo.common import get_vqa_dataset
from utils.config import OUTPUT_DIR, VQA_DATASET_DIR
from utils.logger import Logger

logger = Logger(__name__)

random.seed(42)


class VQADataset(Dataset):
    def __init__(self, config, dataset_name: str, split: str = "val", processor=None, prompt_handler=None, **kwargs):
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.dataset = get_vqa_dataset(dataset_name, split, config.task_type == "multiple_choice")
        self.processor = processor
        self.prompt_handler = prompt_handler
        self.model_name = kwargs.get("model_name")

        self.additional_context_data = {}
        if prompt_handler is not None and self.config.vqa_format in [
            "caption_vqa",
            "cot_vqa",
        ]:  # prompt_handler tell us whether it is for vqa inference or caching the context data
            self.additional_context_data = self.load_cached_context_data(split)

        self.in_context_provider = None
        if "knn" in self.config.prompt_name:
            self.in_context_provider = IncontextExamplesProvider(self.config)

    def get_image_qa_multiple_choice_dataset(self, split, shuffle=False):
        dataset_root = "datasets"
        fname = os.path.join(VQA_DATASET_DIR, dataset_root, self.dataset_name, "dataset.json")
        dataset = json.load(open(fname, "r"))
        dataset_split = defaultdict(list)
        for img in dataset["images"]:
            dataset_split[img["split"]].append(img)

        dataset = {}
        for i, img in enumerate(dataset_split[split]):
            for pair in img["qa_pairs"]:
                qa_obj = {}
                qa_obj["image_id"] = pair["image_id"]
                qa_obj["question"] = pair["question"]
                qa_obj["question_id"] = pair["qa_id"]
                qa_obj["answer"] = pair["answer"]
                qa_obj["mc"] = [pair["answer"]] + pair["multiple_choices"]
                if shuffle:
                    random.shuffle(qa_obj["mc"])
                qa_obj["mc_selection"] = qa_obj["mc"].index(pair["answer"])
                dataset[qa_obj["question_id"]] = qa_obj
        return dataset

    def get_cached_context_file_path(self, split) -> str:
        required_args = [
            self.config.dataset_name,
            self.config.gen_model_name,
            self.config.vqa_format,
            self.config.prompt_name,
        ]
        if any(val is None for val in required_args):
            raise ValueError(
                f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
                f"Provided: {required_args}"
            )
        subdir_prefix = self.config.prompt_name.split(",")[1]
        context_dir_str = (
            "generated_caption_dumps" if self.config.vqa_format == "caption_vqa" else "generated_rationale_dumps"
        )
        dir_path = os.path.join(
            OUTPUT_DIR,
            "cache",
            context_dir_str,
            self.config.dataset_name,
            self.config.gen_model_name,
            self.config.vqa_format,
            subdir_prefix,
            split,
        )
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"output.json")
        return file_path

    def load_cached_context_data(self, split: str):
        fname = self.get_cached_context_file_path(split)
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"File {fname} does not exist. Please generate the prompted captions first.\n"
                f"python caption_generation.py --dataset_name {self.config.dataset_name} "
                f"--model_name {self.config.model_name} --vqa_format caption_qa "
                f"--prompt_name {self.config.prompt_name}"
            )
        logger.info(f"Loading prompted captions from {fname}")
        with open(fname, "r") as f:
            prompted_captions = json.load(f)
        logger.info(f"Loaded prompted captions from {fname}")
        cached_data = prompted_captions
        return cached_data

    def load_cached_context_data_by_ids(self, ids: Union[str, List[str]]):
        if isinstance(ids, str):
            context_batch = self.additional_context_data[str(ids)]  # handling winoground case
        else:
            context_batch = [self.additional_context_data[str(idx)] for idx in ids]
        return context_batch

    def __repr__(self) -> str:
        return f"{self.dataset_name} {self.split} dataset with {len(self)} examples"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        content = self.dataset[idx]
        question_id = content["question_id"]
        image_path = content["image_path"]

        if self.dataset_name in ["vqa_v2", "okvqa"]:
            answer = content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
            question = content["question"]
        else:
            question = content["question"]
            answer = content["answer"]

        if self.dataset_name in ["visual7w", "aokvqa"] and self.config.task_type == "multiple_choice":
            choice = content["choice"]
            choice = [c.strip(punctuation) for c in choice]
            random.shuffle(choice)
            lst_choice = choice.pop()  # remove the last element and store it in a separate variable
            choice = ", ".join(choice) + " or " + lst_choice + "?"

        image = Image.open(image_path).convert("RGB")
        data = {
            "image": image,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "question_id": question_id,
        }

        if self.prompt_handler is not None:
            inp = {"question": question, "add_special_tokens": "flamingo" in self.model_name}
            if self.dataset_name in ["visual7w", "aokvqa"] and self.config.task_type == "multiple_choice":
                inp.update({"choice": choice})

            # TODO: this only works for standard-qa setting
            if self.additional_context_data:
                additional_context_str = self.load_cached_context_data_by_ids(str(question_id))
                context_data_key = "rationale" if self.config.vqa_format == "cot_qa" else "caption"
                inp.update({context_data_key: additional_context_str})

            prompted_question = self.prompt_handler.generic_prompting_handler_for_vqa(inp)

            if self.in_context_provider is not None:
                support_examples = self.in_context_provider.get_support_examples_by_question_id(question_id)
                prompted_question = f"{support_examples} {prompted_question}"

            if "opt" in self.config.model_name:
                prompted_question = prompted_question.replace("\n", " ")

            if "kosmos" in self.config.model_name:
                prompted_question = f"<grounding> {prompted_question}"

            prompted_question = prompted_question.strip()
            data.update({"prompted_question": prompted_question})

        return data


def load_caption(config, split, prompt_name="prefix_promptcap"):
    fpath = os.path.join(
        OUTPUT_DIR,
        "cache",
        "generated_caption_dumps",
        config.dataset_name,
        config.gen_model_name,
        config.vqa_format,
        prompt_name,
        split,
        "output.json",
    )
    caption_data = json.load(open(fpath))
    logger.info(f"Nearest neighbour search > Loaded {len(caption_data)} examples from {fpath}")
    return caption_data


def load_rationale(config, split, prompt_name):
    logger.info(config.dataset_name, config.model_name, prompt_name, split)
    fpath = os.path.join(
        OUTPUT_DIR,
        "cache",
        "generated_rationale_dumps",
        config.dataset_name,
        config.model_name,
        config.vqa_format,
        prompt_name,
        split,
        "output.json",
    )
    rationale_data = json.load(open(fpath))

    return rationale_data


class IncontextExamplesProvider:
    def __init__(self, config):
        self.config = config
        self.split = "train"
        if "," in config.prompt_name:
            self.prompt_name = config.prompt_name.split(",")[1]
        else:
            self.prompt_name = config.prompt_name
        self.dataset_name = config.dataset_name
        self.vqa_format = config.vqa_format
        self.support_data_dict = self.get_support_data_dict(
            config.dataset_name, self.split, self.prompt_name, config.task_type == "multiple_choice"
        )
        self.nearest_neighbor_ids = self.load_cached_nearest_neighbor_ids(config.dataset_name)

    def load_cached_nearest_neighbor_ids(self, dataset_name):
        fpath = os.path.join(
            OUTPUT_DIR,
            "cache",
            "nearest_neighbors",
            dataset_name,
            "train_to_val",
            "output.json",
        )

        nearest_neighbor_ids = json.load(open(fpath))
        return nearest_neighbor_ids

    def get_support_data_dict(self, dataset_name: str, split: str, multiple_choice: bool, prompt_name: str):
        exemplar_dataset = get_vqa_dataset(dataset_name, split, multiple_choice)

        context_data = {}
        if self.vqa_format == "caption_qa":
            context_data = load_caption(self.config, split, prompt_name)
        elif self.vqa_format == "cot_qa":
            context_data = load_rationale(self.config, split, prompt_name)

        support_data_map = {}
        for idx in range(len(exemplar_dataset)):
            content = exemplar_dataset[idx]

            question_id = content["question_id"]
            if self.dataset_name in ["vqa_v2", "okvqa"]:
                answer = content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
                question = content["question"]
            else:
                question = content["question"]
                answer = content["answer"]

            choices = content.get("choice", "")
            if choices:
                last_choice = choices.pop()
                options = f"{', '.join(choices)} or {last_choice}?"
                choices = options

            context = context_data.get(str(question_id), "")

            support_data_map[question_id] = {
                "question": question,
                "answer": answer,
                "choice": choices,
                "context": context,
            }
        return support_data_map

    # TODO: this is a hacky way to add the support examples to the data, need to refactor this
    def format_support_example(self, support_ex, vqa_format):
        if vqa_format in ["caption_vqa", "standard_vqa"]:
            context = support_ex.get("context", "")
            context_str = f"{context} " if context else ""
            if "short_answer" in self.config.prompt_name:
                return f"Context: {context_str} Question: {support_ex['question']} {support_ex.get('choice', '')} Short answer: {support_ex['answer']} "
            else:
                return f" {context_str} {support_ex['question']} {support_ex.get('choice', '')} {support_ex['answer']}"
        elif vqa_format == "cot_vqa":
            return f"Q: {support_ex.get('question', '')} A: {support_ex['context']}"
        else:
            raise ValueError(f"Invalid vqa_format: {vqa_format}")

    def get_support_examples_by_question_id(self, question_id):
        support_question_ids = self.nearest_neighbor_ids.get(
            question_id, self.nearest_neighbor_ids.get(str(question_id))
        )
        support_examples = [
            self.support_data_dict.get(qid, self.support_data_dict.get(str(qid))) for qid in support_question_ids
        ]
        logger.debug(f"Support examples: {support_examples}")
        support_examples = [self.format_support_example(ex, self.vqa_format) for ex in support_examples]
        random.shuffle(support_examples)
        support_questions = "\n".join(support_examples)
        instructions = "In this task, your goal is to write an answer to a question about the image.\n---\nTo write the answer, here are some sample QA suggestions (not related to the given image):\n"
        # instructions = "In this task, your goal is to write an answer to a question about the image. Here are some suggested QA pairs (not related to the given image)\n"
        support_questions = instructions + support_questions + "\nNow answer the following question about the image."
        return support_questions


def collate_fn_builder(processor=None, tokenizer=None):
    def collate_fn(batch):
        # get the keys from the first batch element
        batch_keys = batch[0].keys()
        bkeys = ["question", "answer", "question_id", "prompted_question", "image", "image_path"]

        # Create the batch dictionary
        processed_batch = {}
        for bkey in bkeys:
            if bkey in batch_keys:
                processed_batch[bkey + "s"] = [example[bkey] for example in batch]

        if processor is not None:
            encoding = processor(
                images=processed_batch["images"],
                text=processed_batch["prompted_questions"],
                padding=True,
                return_tensors="pt",
            )
            processed_batch.update(**encoding)

        if tokenizer is not None:
            text_inputs = tokenizer(
                [example["prompted_question"] for example in batch], padding=True, return_tensors="pt"
            )
            processed_batch["input_ids"] = text_inputs["input_ids"]
            processed_batch["attention_mask"] = text_inputs["attention_mask"]
        # processed_batch["pixel_values"] = torch.stack([example["pixel_values"] for example in batch])
        return processed_batch

    return collate_fn


def collate_fn(batch):
    # get the keys from the first batch element
    batch_keys = batch[0].keys()
    bkeys = ["question", "answer", "question_id", "prompted_question", "image", "image_path"]
    # Create the batch dictionary
    batch_dict = {}
    for bkey in bkeys:
        if bkey in batch_keys:
            batch_dict[bkey + "s"] = [example[bkey] for example in batch]

    return batch_dict
