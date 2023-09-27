import json
import os
import re
import time
from typing import List

import torch
from PIL import Image
from promptcap import PromptCap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForVision2Seq, AutoProcessor,
                          Blip2ForConditionalGeneration)

from dataset_zoo.custom_dataset import VQADataset, collate_fn
from utils.config import OUTPUT_DIR
from utils.globals import MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.logger import Logger
from vqa_zero.inference_utils import (apply_prompt_to_example_winoground,
                                      get_image_tensors_winoground,
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

    def load_model_and_processor(self):
        model_name = MODEL_CLS_INFO["hfformer"][self.args.gen_model_name]["name"]
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

        if "kosmos" in model_name:
            model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
            model = model.to("cuda")
            if "kosmos" in model_name:
                processor.tokenizer.padding_side = "left"
        else:
            if "opt" in model_name:
                processor.tokenizer.padding_side = "left"
            model = Blip2ForConditionalGeneration.from_pretrained(
                model_name, torch_dtype=torch.bfloat16, device_map="auto"
            )
        logger.info(
            f"Initialized {self.__class__.__name__} with model '{model.__class__.__name__}' and processor '{processor.__class__.__name__}'"
        )

        return model, processor

    def generate_caption(self, prompt_handler: PromptingHandler, split="val"):
        if os.path.exists(self.get_file_path(split)):
            logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path(split)}")
            return

        self.model, self.processor = self.load_model_and_processor()
        batch_size = 32  # hardcoded for now
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
            # "processor": self.processor,
            # "prompt_handler": prompt_handler,
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
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )

        logger.info(f" Num examples: \t{len(dataset)}")
        logger.info(f" Batch size: \t{batch_size}")
        logger.info(f" Num iterations: \t{len(dataset) // batch_size}")

        output = {}
        for batch in tqdm(dataloader, desc="Generating captions"):
            images = batch["images"]
            questions = batch["questions"]
            answers = batch["answers"]
            prompts = self.get_prompt_str_for_prefix_caption(prompt_handler, questions)  # prefix for caption
            # inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(
            #     self.device, torch.bfloat16
            # )
            # generated_ids = self.model.generate(
            #     **inputs,
            #     max_new_tokens=100,
            #     num_beams=3,
            #     num_return_sequences=1,
            #     no_repeat_ngram_size=3,
            # )
            # generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            generated_captions = self._generate(prompts, images)

            generated_captions = [re.sub(r"^\s+|\s+$", "", caption) for caption in generated_captions]
            if len(generated_captions) != len(questions):
                new_list = []
                for i in range(0, len(generated_captions), 3):
                    current_caption_list = generated_captions[i : i + 3]
                    logger.info(current_caption_list)
                    # take the maximum length caption
                    current_caption = max(current_caption_list, key=len)
                    new_list.append(current_caption)
                generated_captions = new_list

            # these are fixed streps for all caption generation
            if prompt_handler.prompt_name == "prefix_a_photo_of":
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            elif not prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [
                    prompt_handler.generic_prompting_handler_for_vqa({"caption": ex}) for ex in generated_captions
                ]

            if len(generated_captions) != len(questions):
                new_list = []
                for i in range(0, len(generated_captions), 3):
                    new_list.append("".join(generated_captions[i : i + 3]))
                generated_captions = new_list
            assert len(generated_captions) == len(questions)

            for question_id, caption in zip(batch["question_ids"], generated_captions):
                output[question_id] = caption.replace("\n", "").strip()

            for question, caption, answer in zip(questions, generated_captions, answers):
                logger.info(f"Question = {question}, Caption = {caption}, Answer = {answer}")

        self.save(output, split)

    def generate_blip2(self, inputs, max_new_tokens=50, num_beams=3, length_penalty=1.4, no_repeat_ngram_size=3):
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                length_penalty=length_penalty,
                num_beams=num_beams,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        generated_outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_outputs

    def generate_kosmos(self, inputs, max_new_tokens=150):
        with torch.no_grad():
            generated_ids = self.model.generate(
                pixel_values=inputs["pixel_values"].to("cuda"),
                input_ids=inputs["input_ids"][:, :-1].to("cuda"),
                attention_mask=inputs["attention_mask"][:, :-1].to("cuda"),
                img_features=None,
                img_attn_mask=inputs["img_attn_mask"][:, :-1].to("cuda"),
                use_cache=True,
                max_new_tokens=max_new_tokens,
            )
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
            output = self.processor[1].batch_decode(generated_ids, skip_special_tokens=True)

        return output

    def _generate(self, prompts: List[str], images: List[Image.Image]):
        if self.args.gen_model_name.startswith("blip"):
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(
                self.device, torch.bfloat16
            )
            generated_captions = self.generate_blip2(inputs)

        elif self.args.gen_model_name.startswith("kosmos"):
            prompts = [f"<grounding> {text}" for text in prompts]
            inputs = self.processor(images=images, text=prompts, padding=True, return_tensors="pt").to(self.device)
            generated_captions = self.generate_kosmos(inputs)
            prompts = [re.sub("^<grounding> ", "", q) for q in prompts]
            generated_captions = [re.split(r"[\n.]", o[len(q) :])[0] for q, o in zip(prompts, generated_captions)]
            generated_captions = [o.strip() for o in generated_captions]

        elif self.args.gen_model_name == "open_flamingo_lamma":
            from mlfoundations.utils import get_demo_image

            if isinstance(self.processor, tuple):
                vis_processors, txt_processors = self.processor
            demp_vision_x = get_demo_image(self.args, vis_processors, self.device)
            cap_image_tensors = torch.cat(image_tensors, dim=0).unsqueeze(1).unsqueeze(1).to(self.device)
            image_tensors = torch.cat((demp_vision_x, cap_image_tensors), dim=1)

            few_show_examples = (
                f"<image>A photo of two cats dozing off on a pink sofa, with two remotes lying nearby. Both cats has the same color, with shades of brown and black.<|endofchunk|>"
                f"<image>A photo of a bathroom featuring a pristine white sink with a mirror resting atop it. A white curtain hangs in the left corner, while toiletry items can be seen placed on the nearby shelf.<|endofchunk|>"
                f"<image>A photo of two cats dozing off on a pink sofa, with two remotes lying nearby. Both cats has the same color, with shades of brown and black.<|endofchunk|>"
                f"<image>A photo of a bathroom featuring a pristine white sink with a mirror resting atop it. A white curtain hangs in the left corner, while toiletry items can be seen placed on the nearby shelf.<|endofchunk|>"
            )

            prompted_input_text = [few_show_examples + "<image>A photo of"] * len(image_tensors)
            prompted_input_text = [few_show_examples + "<image>A photo of"] * len(image_tensors)
            prompted_lang_x = txt_processors(prompted_input_text, return_tensors="pt", padding=True).to(self.device)
            generated_captions = self.generate_flamingo(image_tensors, prompted_lang_x)
            few_show_examples = few_show_examples.replace("<|endofchunk|>", " ").replace("<image>", "")
            logger.info(f"few_show_examples: {few_show_examples}")
            generated_captions = [text.replace(few_show_examples, "") for text in generated_captions]
            logger.info(f"Generated captions: {json.dumps(generated_captions, indent=2)}")
        else:
            inputs = self.processor(text=prompts, images=images, add_special_tokens=False).to(self.device)
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


class CaptionPrefixHandlerWinoground(CaptionGenerator):
    NUM_IMAGES = 2

    def batchify_winoground(self, winoground_example):
        questions = [winoground_example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
        images = [winoground_example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
        batch = {"questions": questions, "images": images}
        batch["question_ids"] = winoground_example["id"]
        return batch

    def generate_caption(self, dataloader, prompt_handler: PromptingHandler):
        fname = self.get_file_path()
        if os.path.exists(fname):
            file_mod_time = os.path.getmtime(fname)
            current_time = time.time()
            two_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < two_days_in_seconds:
                logger.info(f"Caption data already exists. You can load it from cache {fname}")
                return

        self.model, self.processor = load_model_and_processors(self.args.model_name, self.args.device)

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        output = {}
        for example in tqdm((dataloader)):
            caption_texts = [example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
            image_tensors = get_image_tensors_winoground(example, self.processor)
            if prompt_handler.prompt_name == "long_caption":
                generated_captions = self.generate_pipelined(image_tensors, caption_texts)
            elif prompt_handler.prompt_name == "prefix_iterative_caption":
                generated_captions = self.generate_iterative_captions(image_tensors, caption_texts)
            else:
                generated_captions = self.generate_standard(image_tensors, caption_texts, prompt_handler)

            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                caption_example = {f"caption_{i}": generated_captions[i] for i in range(self.NUM_IMAGES)}
                generated_captions = apply_prompt_to_example_winoground(prompt_handler, caption_example)

            output[example["id"]] = generated_captions
        self.save(output, split="val")

    def generate_standard(self, image_tensors, questions: List[str], prompt_handler: PromptingHandler):
        prompt_txt = self.get_prompt_str_for_prefix_caption(prompt_handler, questions)  # prefix for caption

        if self.args.model_name.startswith("blip"):
            samples = {
                "image": torch.stack(image_tensors).to(self.model.device),
                "prompt": prompt_txt,
            }
            generated_captions = self.generate_blip(samples)
        else:
            input_ids = self.processor(text=prompt_txt, add_special_tokens=False).input_ids
            samples = {
                "input_ids": input_ids,
                "pixel_values": torch.stack(image_tensors).to(self.model.device),
            }
            generated_captions = self.generate_hfformer(
                samples, max_new_tokens=50, num_beams=4, length_penalty=0.6, no_repeat_ngram_size=3
            )
        return generated_captions


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
        for batch in tqdm((dataloader), desc="Generating captions"):
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
            output = self.model.caption(prompt, image)
            generated_captions.append(output)
        return generated_captions
