import json
import os
import re
import time
from typing import List

import torch
from datasets import load_dataset
from PIL import Image
from promptcap import PromptCap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForVision2Seq, AutoProcessor,
                          Blip2ForConditionalGeneration)

from dataset_zoo.custom_dataset import VQADataset, collate_fn, collate_fn_builder
from utils.config import OUTPUT_DIR
from utils.globals import MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.logger import Logger
from vqa_zero.inference_utils import (apply_prompt_to_example_winoground,
                                      load_model_and_processors)

logger = Logger(__name__)

N = 5  # number of days to keep the cached caption data


class CaptionGenerator:
    """
    This class is used to generate captions for the images in the VQA dataset.
    """

    def __init__(self, args, device="cuda"):
        self.args = args
        self.data = None  # caption data
        self.device = device

    def _initialize_dataloader(self, prompt_handler, processor, tokenizer, split):
        batch_size = 32  # hardcoded for now
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
            "prompt_handler": prompt_handler,
            "model_name": self.args.model_name,
            "split": split,
        }
        dataset = VQADataset(**dataset_args)
        collate_fn = collate_fn_builder(processor, tokenizer)

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.args.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
        )

    def generate_caption(self, prompt_handler: PromptingHandler, split="val"):
        if os.path.exists(self.get_file_path(split)):
            logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path(split)}")
            return

        self.model, self.processor, self.tokenizer = load_model_and_processors(
            self.args.model_name, self.args.device, self.args.autocast_dtype
        )
        dataloader = self._initialize_dataloader(prompt_handler, self.processor, self.tokenizer, split)

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )

        logger.info(f" Num examples: \t{len(dataloader.dataset)}")
        logger.info(f" Batch size: \t{dataloader.batch_size}")
        logger.info(f" Num iterations: \t{len(dataloader.dataset) // dataloader.batch_size}")

        output = {}
        for batch in tqdm(dataloader, desc="Generating captions"):
            questions = batch["question"]
            answers = batch["answer"]

            generated_captions = self._generate(batch)
            generated_captions = self.apply_formatting(generated_captions, questions)

            # these are fixed streps for all caption generation
            if prompt_handler.prompt_name == "prefix_a_photo_of":
                generated_captions = [f"A photo of {gen_caption}" for gen_caption in generated_captions]
            elif not prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [
                    prompt_handler.generic_prompting_handler_for_vqa({"caption": ex}) for ex in generated_captions
                ]

            for question_id, caption in zip(batch["question_id"], generated_captions):
                output[question_id] = caption.replace("\n", "").strip()

            for question, caption, answer in zip(questions, generated_captions, answers):
                logger.info(f"Question = {question}, Caption = {caption}, Answer = {answer}")

        self.save(output, split)

    def apply_formatting(self, generated_captions, questions):
        generated_captions = [re.sub(r"^\s+|\s+$", "", caption) for caption in generated_captions]
        generated_captions = [o + "." if not o.endswith(".") else o for o in generated_captions]
        if len(generated_captions) != len(questions):
            new_list = []
            for i in range(0, len(generated_captions), 3):
                new_list.append(" ".join(generated_captions[i : i + 3]))
            generated_captions = new_list
            
            assert len(generated_captions) == len(questions)

        return generated_captions

    def generate_blip2(self, inputs, max_new_tokens=150):
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                top_p=0.8,
                num_return_sequences=3,
            )
        generated_outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_outputs

    def generate_kosmos(self, inputs, max_new_tokens=256):
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"].to("cuda"),
            input_ids=inputs["input_ids"][:, :-1].to("cuda"),
            attention_mask=inputs["attention_mask"][:, :-1].to("cuda"),
            img_features=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1].to("cuda"),
            use_cache=True,
            max_new_tokens=max_new_tokens,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] - 1 :]
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_outputs, entities_list = zip(*[self.processor.post_process_generation(gt) for gt in generated_texts])

        return generated_outputs

    def generate_hfformer(self, inputs, max_new_tokens=50, num_beams=5, length_penalty=1.4, no_repeat_ngram_size=3):
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return output

    def generate_flamingo(self, prompted_image_tensors, cap_lang_x):
        with torch.no_grad(), torch.cuda.amp.autocast():
            generated_ids = self.model.generate(
                vision_x=prompted_image_tensors,
                lang_x=cap_lang_x["input_ids"],
                attention_mask=cap_lang_x["attention_mask"],
                max_new_tokens=100,
                num_beams=5,
                length_penalty=1,
                top_p=0.9,
                num_return_sequences=1,
            )
            output = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return output
    
    def generate_llava(self, batch):
        from LLaVA.llava.conversation import (SeparatorStyle,
                                                conv_templates)
        from LLaVA.llava.mm_utils import KeywordsStoppingCriteria

        conv = conv_templates[self.processor.conv_mode].copy()
        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = (
            [KeywordsStoppingCriteria(keywords, self.processor.tokenizer, input_ids)]
            if conv.version == "v0"
            else None
        )
        input_ids = batch["input_ids"]
        image_tensor = batch["image_tensors"]
        input_ids = input_ids.cuda()

        output_ids = self.model.generate(
            input_ids,
            images=image_tensor.half().cuda(),
            do_sample=False,
            num_beams=self.args.num_beams,
            max_new_tokens=self.args.max_length,
            length_penalty=self.args.length_penalty,
            use_cache=True,
            stopping_criteria=stopping_criteria,
        )
        generated_outputs = self.processor.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1] :], skip_special_tokens=True
        )
        generated_outputs = [out.strip() for out in generated_outputs]
        generated_outputs = [
            out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs
        ]

        return generated_outputs

    def _generate(self, batch):
        if self.args.gen_model_name.startswith("blip"):
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "pixel_values": batch["pixel_values"].to(self.device, self.args.autocast_dtype),
            }
            generated_captions = self.generate_blip2(inputs)

        elif self.args.gen_model_name.startswith("kosmos"):
            generated_captions = self.generate_kosmos(batch)

        elif self.args.gen_model_name == "open_flamingo_lamma":            
            generated_captions = self.generate_flamingo(inputs) 
            logger.info(f"Generated captions: {json.dumps(generated_captions, indent=2)}")

        elif self.args.model_name == "llava":
            generated_captions = self.generate_llava(inputs)

        else:
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "pixel_values": batch["pixel_values"].to(self.device, self.args.autocast_dtype),
            }
            generated_captions = self.generate_hfformer(
                **inputs, max_new_tokens=50, num_beams=3, length_penalty=1.4, no_repeat_ngram_size=3
            )

        return generated_captions

    def save(self, output_captions, split):
        fname = self.get_file_path(split)
        with open(fname, "w") as f:
            json.dump(output_captions, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved generated captions to {fname}")

    def get_file_path(self, split) -> str:
        required_args = [self.args.dataset_name, self.args.gen_model_name, self.args.vqa_format, self.args.prompt_name]
        if any(val is None for val in required_args):
            raise ValueError(
                f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
                f"Provided: {required_args}"
            )
        subdir_prefix = self.args.prompt_name.split(",")[1]
        # TODO: design a better way to handle this
        dir_path = os.path.join(
            OUTPUT_DIR,
            "cache",
            "generated_caption_dumps",
            self.args.dataset_name,
            self.args.gen_model_name,  # "git_large_textcaps",
            self.args.vqa_format,
            subdir_prefix,
            split,
        )
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(dir_path, f"output.json")
        return file_path

    @staticmethod
    def get_prompt_str_for_prefix_caption(prompt_handler: PromptingHandler, questions: List[str]) -> List[str]:
        if "promptcap" in prompt_handler.prompt_name:
            prompt_txt = []
            for q in questions:
                prompt_txt.append(prompt_handler.prompt.apply({"question": q})[0])
        elif prompt_handler.prompt_name.startswith("prefix_"):
            prompt_txt = prompt_handler.prompt.apply({})[0]
            logger.debug(f"PROMPT FOR CAPTION GENERATION: {prompt_txt}")
        else:
            prompt_txt = ""

        if isinstance(prompt_txt, str):
            prompt_txt = [prompt_txt] * len(questions)
        return prompt_txt


class CaptionGeneratorWinoground(CaptionGenerator):
    NUM_IMAGES = 2

    def batchify_winoground(self, winoground_example):
        questions = [winoground_example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
        images = [winoground_example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
        batch = {"questions": questions, "images": images}
        batch["question_ids"] = winoground_example["id"]
        return batch

    def generate_caption(self, prompt_handler: PromptingHandler):
        fname = self.get_file_path(split="test")
        if os.path.exists(fname):
            logger.info(f"Caption data already exists. You can load it from cache {fname}")
            return

        dataloader = load_dataset("facebook/winoground", use_auth_token=True)["test"]
        self.model, self.processor, _ = load_model_and_processors(
            self.args.model_name, self.args.device, self.args.autocast_dtype
        )

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        output = {}
        for example in tqdm(dataloader):
            caption_texts = [example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
            images = [example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
            prompts = self.get_prompt_str_for_prefix_caption(prompt_handler, caption_texts)  # prefix for caption

            generated_captions = self._generate(prompts, images)

            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                caption_example = {f"caption_{i}": generated_captions[i] for i in range(self.NUM_IMAGES)}
                generated_captions = apply_prompt_to_example_winoground(prompt_handler, caption_example)

            output[example["id"]] = generated_captions

        self.save(output, split="test")


class PromptCapGenerarator(CaptionGenerator):
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device

    def _load_model(self, device):
        self.promptcap = PromptCap("vqascore/promptcap-coco-vqa")  # Load the PromptCap model
        self.promptcap.model.to(device)

    def generate_caption(self, prompt_handler: PromptingHandler, split):
        if os.path.exists(self.get_file_path(split)):
            logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path(split)}")
            return

        self._load_model(self.device)

        batch_size = 32  # hardcoded for now
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
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
            f" Generating caption data for {self.args.dataset_name} dataset, {split} split."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        logger.info(f" Num examples: \t{len(dataset)}")
        logger.info(f" Batch size: \t{batch_size}")
        logger.info(f" Num iterations: \t{len(dataset) // batch_size}")

        output = {}
        for batch in tqdm(dataloader, desc="Generating captions"):
            images = batch["image"]
            questions = batch["question"]

            prompted_questions = [
                f"Please describe this image according to the given question: {question}" for question in questions
            ]
            generated_captions = self.promptcap.caption_batch(prompted_questions, images)

            # these are fixed streps for all caption generation
            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                generated_captions = [
                    prompt_handler.generic_prompting_handler_for_vqa({"caption": ex}) for ex in generated_captions
                ]
            logger.debug(f"Generated captions: {json.dumps(generated_captions)}")
            assert len(generated_captions) == len(questions)

            for question_id, caption in zip(batch["question_id"], generated_captions):
                output[question_id] = caption

        self.save(output, split)

    def generate_standard(self, images, questions):
        generated_captions = []
        for image, question in zip(images, questions):
            prompt = f"Please describe this image according to the given question: {question}"
            output = self.promptcap.caption(prompt, image)
            generated_captions.append(output)
        return generated_captions


class PromptCapGeneratorWinoground(PromptCapGenerarator):
    NUM_IMAGES = 2

    def batchify_winoground(self, winoground_example):
        questions = [winoground_example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
        images = [winoground_example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
        batch = {"questions": questions, "images": images}
        batch["question_ids"] = winoground_example["id"]
        return batch

    def generate_caption(self, prompt_handler: PromptingHandler):
        fname = self.get_file_path(split="test")
        if os.path.exists(fname):
            logger.info(f"Caption data already exists. You can load it from cache {fname}")
            return

        self._load_model(self.device)

        dataloader = load_dataset("facebook/winoground", use_auth_token=True)["test"]

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )

        output = {}
        for example in tqdm(dataloader):
            images = [example[f"image_{i}"] for i in range(self.NUM_IMAGES)]
            cached_questions = prompt_handler.load_cached_questions_for_winoground(example["id"])

            generated_captions = self.generate_standard(images, cached_questions)

            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                caption_example = {f"caption_{i}": generated_captions[i] for i in range(self.NUM_IMAGES)}
                generated_captions = apply_prompt_to_example_winoground(prompt_handler, caption_example)

            output[example["id"]] = generated_captions
        self.save(output, split="test")

    def generate_standard(self, images, questions):
        generated_captions = []
        for image, question in zip(images, questions):
            prompt = f"Please describe this image according to the given question: {question}"
            output = self.promptcap.caption(prompt, image)
            generated_captions.append(output)
        return generated_captions
