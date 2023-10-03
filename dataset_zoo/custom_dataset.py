import json
import os
import random
from string import punctuation
from typing import Dict, List, Union

from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from dataset_zoo.common import get_vqa_dataset
from utils.config import OUTPUT_DIR
from utils.handler import PromptingHandler
from utils.logger import Logger
from vqa_zero.inference_utils import get_prompt_template_handler

logger = Logger(__name__)

random.seed(42)


class VQADataset(Dataset):
    def __init__(
        self,
        config,
        dataset_name: str,
        split: str = "val",
        prompt_handler: Union[PromptingHandler, None] = None,
        **kwargs,
    ):
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.dataset = get_vqa_dataset(dataset_name, split, config.task_type == "multiple_choice")
        self.prompt_handler = prompt_handler
        self.model_name = kwargs.get("model_name")
        self.additional_context_data = {}
        self.demo_samples_provider = None

        if prompt_handler:
            self._initialize_cache_on_prompt_handler()

    def _initialize_cache_on_prompt_handler(self):
        if self.prompt_handler.subset_name == "vqa" and self.config.vqa_format in ["caption_vqa", "cot_vqa"]:
            self.additional_context_data = self.load_cached_context_data(self.split)

        if self.config.few_shot:
            self.demo_samples_provider = DemoSamplesProvider(self.config)

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
        data = {
            "image": Image.open(content["image_path"]).convert("RGB"),
            "image_path": content["image_path"],
            "question": content["question"],
            "answer": self._get_answer(content),
            "question_id": content["question_id"],
        }

        if self.prompt_handler:
            data.update(self._get_prompted_data(content))
        return data

    def _get_answer(self, content):
        if self.dataset_name in ["vqa_v2", "okvqa"]:
            return content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
        return content["answer"]

    def _get_prompted_data(self, content):
        data = {}
        question = content["question"]
        if self.prompt_handler.subset_name == "vqa":
            inp = {"question": question}
            if self.dataset_name in ["visual7w", "aokvqa"] and self.config.task_type == "multiple_choice":
                inp.update({"choice": self._get_choice(content)})

            if self.additional_context_data:
                additional_context_str = self.additional_context_data[str(content["question_id"])]
                if additional_context_str[-1] != ".":
                    additional_context_str += "."
                context_data_key = "rationale" if self.config.vqa_format == "cot_vqa" else "caption"
                inp.update({context_data_key: additional_context_str})

            prompted_question = self.prompt_handler.generic_prompting_handler_for_vqa(inp)
            data["prompted_question_without_demo"] = prompted_question

            if self.demo_samples_provider:
                support_data = self.demo_samples_provider.get_support_examples_by_question_id(content["question_id"])
                data["support_examples"] = support_data["support_examples"]
                prompted_question = support_data["formatted_support_text"] + prompted_question
            if "opt" in self.config.model_name:
                prompted_question = prompted_question.replace("\n", " ")

            # if "kosmos" in self.config.model_name:
            #     prompted_question = f"<grounding> {prompted_question}"
            data["prompted_question"] = prompted_question.strip()

        elif self.prompt_handler.subset_name == "captioning":
            prompted_question = self._get_prompt_str_for_single_question(question)
            data.update({"prompted_question": prompted_question, "prompted_question_without_demo": prompted_question})

        return data

    def _get_choice(self, content):
        choice = content["choice"]
        choice = [c.strip(punctuation) for c in choice]
        random.shuffle(choice)
        lst_choice = choice.pop()
        return ", ".join(choice) + " or " + lst_choice + "?"

    def _get_prompt_str_for_single_question(self, question: str) -> str:
        """
        Get prompt string for a single question.

        Parameters:
        - prompt_handler: The handler object for prompting.
        - question (str): A single question string.

        Returns:
        str: The generated prompt string.
        """
        if "promptcap" in self.prompt_handler.prompt_name:
            prompt_txt = self.prompt_handler.prompt.apply({"question": question})[0]
        elif self.prompt_handler.prompt_name.startswith("prefix_"):
            prompt_txt = self.prompt_handler.prompt.apply({})[0]
            logger.debug(f"PROMPT FOR CAPTION GENERATION: {prompt_txt}")
        else:
            prompt_txt = ""

        return prompt_txt


class DemoSamplesProvider:
    def __init__(self, config):
        self.config = config
        self.split = "train_bal" if config.dataset_name == "gqa" else "train"
        self.dataset_name = config.dataset_name
        self.vqa_format = config.vqa_format
        prompt_handler, _ = get_prompt_template_handler(config, config.prompt_name)
        if not isinstance(prompt_handler, list):
            self.fewshot_prompt_handler = prompt_handler
            context_prompt_name = None
        else:
            self.fewshot_prompt_handler = prompt_handler[0]
            context_prompt_name = prompt_handler[1].prompt_name

        self.support_data_dict = self.load_support_data(
            config.dataset_name, self.split, config.task_type == "multiple_choice", context_prompt_name
        )
        self.nearest_neighbor_ids = self.load_nearest_neighbor_ids_from_cache(config.dataset_name)

    def load_nearest_neighbor_ids_from_cache(self, dataset_name):
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

    def load_support_data(self, dataset_name: str, split: str, multiple_choice: bool, prompt_name: str):
        logger.info(
            f"Loading support data for name = {dataset_name}, split = {split}, prompt = {prompt_name} and multiple_choice = {multiple_choice}"
        )
        exemplar_dataset = get_vqa_dataset(dataset_name, split, multiple_choice)

        context_data = {}
        if self.vqa_format == "caption_vqa":
            context_data = self.load_generated_captions(self.config, split, prompt_name)

        elif self.vqa_format == "cot_vqa":
            context_data = self.load_generated_rationales(self.config, split, prompt_name)

        support_data_map = {}
        for content in tqdm(exemplar_dataset, desc="Loading support data"):
            question_id = content["question_id"]
            answer = self._get_answer(content)
            answer = answer + "."  # add a period at the end of the answer
            context = context_data.get(question_id, context_data.get(str(question_id)))

            support_data_map[question_id] = {
                "question": content["question"],
                "answer": answer,
                "choice": self._get_choice(content),
                "caption": context,
                "image_path": content["image_path"],
            }
        return support_data_map

    def get_support_examples_by_question_id(self, question_id):
        support_question_ids = self.nearest_neighbor_ids.get(
            question_id, self.nearest_neighbor_ids.get(str(question_id))
        )
        support_data = [
            self.support_data_dict.get(qid, self.support_data_dict.get(str(qid))) for qid in support_question_ids
        ]
        logger.debug(f"Support examples: {support_data}")

        support_examples = [
            self.fewshot_prompt_handler.generic_prompting_handler_for_vqa({**ex, "fewshot": True})
            for ex in support_data
        ]
        for ex_dict, ex_fmt in zip(support_data, support_examples):
            ex_dict["prompted_question"] = ex_fmt

        random.shuffle(support_examples)
        support_questions = ""
        for i, example in enumerate(support_examples, start=1):
            support_questions += f"{i}. {example}\n"
            # support_questions += f"{example}\n"

        logger.debug(f"Support questions: {support_questions}")

        instruction_start = "You task is visual question answering. Here are some suggested QA pairs (not related to the given image):\n"
        instruction_end = "Now, answer the following question about the image.\n"
        # instructions = "In this task, your goal is to write an answer to a question about the image. Here are some suggested QA pairs (not related to the given image)\n"
        formatted_support_text = instruction_start + support_questions + instruction_end

        return {"support_examples": support_data, "formatted_support_text": formatted_support_text}

    def _get_answer(self, content):
        if self.dataset_name in ["vqa_v2", "okvqa"]:
            return content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
        return content["answer"]

    def _get_choice(self, content):
        choice = content.get("choice", [])
        if choice:
            choice = [c.strip(punctuation) for c in choice]
            random.shuffle(choice)
            lst_choice = choice.pop()
            return ", ".join(choice) + " or " + lst_choice + "?"

        return ""

    @staticmethod
    def load_generated_captions(config, split, prompt_name="prefix_promptcap"):
        logger.info(f"Loading generated captions for {config.dataset_name}, {config.model_name}, {split}")

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

    @staticmethod
    def load_generated_rationales(config, split, prompt_name):
        logger.info(f"Loading generated rationales for {config.dataset_name}, {config.model_name}, {split}")
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


def get_flamingo_format(text: str, answer=None) -> str:
    return f"<image>{text} {'' if answer is None else ''}"


def prepare_flamingo_data(example):
    demo_samples = example.get("support_examples")

    if demo_samples:  # num_shots > 0
        context_images = [Image.open(x["image_path"]).convert("RGB") for x in demo_samples]
        context_text = "".join(
            [get_flamingo_format(text=x["prompted_question"], answer=x["answer"]) + "\n" for x in demo_samples]
        )
    else:
        context_images = []
        context_text = ""

    context_images.append(example["image"])
    context_text += get_flamingo_format(text=example["prompted_question_without_demo"])

    if demo_samples is None:
        context_text = context_text.replace("<image>", "")

    return context_images, context_text


def collate_fn_builder(processor=None, tokenizer=None):
    def collate_fn(batch):
        bkeys = ["question", "answer", "question_id", "prompted_question", "image", "image_path"]
        processed_batch = {bkey: [example[bkey] for example in batch] for bkey in bkeys if bkey in batch[0]}

        if processor is not None:
            from dataset_zoo.custom_processor import (FlamingoProcessor,
                                                      LlaVaProcessor)

            if isinstance(processor, FlamingoProcessor):
                batch_images, batch_text = zip(*[prepare_flamingo_data(example) for example in batch])

                processed_batch["image_tensors"] = processor._prepare_images(batch_images)
                input_ids, attention_mask = processor._prepare_text(batch_text)
                processed_batch.update({"input_ids": input_ids, "attention_mask": attention_mask})

            elif isinstance(processor, LlaVaProcessor):
                batch_images, batch_text = processor.get_processed_tokens_batch(
                    [example["prompted_question"] for example in batch], [example["image_path"] for example in batch]
                )

                processed_batch["image_tensors"] = batch_images
                processed_batch["input_ids"] = batch_text

            else:
                encoding = processor(
                    images=processed_batch["image"],
                    text=processed_batch["prompted_question"],
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
            batch_dict[bkey] = [example[bkey] for example in batch]

    return batch_dict
