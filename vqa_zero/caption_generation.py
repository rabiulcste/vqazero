import copy
import json
import os
import re
from typing import List

import torch
from datasets import load_dataset
from PIL import Image
from promptcap import PromptCap
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset_zoo.custom_dataset import (VQADataset, collate_fn,
                                        collate_fn_builder)
from evals.vicuna_llm import LlmEngine
from utils.config import OUTPUT_DIR, PROJECT_DIR
from utils.handler import PromptingHandler
from utils.logger import Logger
from vqa_zero.inference_utils import (apply_prompt_to_example_winoground,
                                      load_model_and_processors)

logger = Logger(__name__)


def set_cuda_visible_devices():
    available_gpus = [i for i in range(torch.cuda.device_count())]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, available_gpus))


set_cuda_visible_devices()


class CaptionGenerator:
    """
    This class is used to generate captions for the images in the VQA dataset.
    """

    def __init__(self, args, device="cuda"):
        self.args = args
        self.data = None  # caption data
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

    def _load_llm_engine(self):
        # check https://vllm.readthedocs.io/en/latest/models/supported_models.html#supported-models for supported models
        model_path = "Open-Orca/Mistral-7B-OpenOrca"
        demonstrations_fname = os.path.join(PROJECT_DIR, "evals", "demonstrations", "dense_captioning.json")
        llm = LlmEngine(model_path, self.device, fname=demonstrations_fname)
        llm.load_pretrained_model_tokenizer()
        llm._load_in_context_examples()

        return llm

    def extract_desired_text(self, text, num_examples_in_task_prompt):
        split_text = text.split(
            f"{num_examples_in_task_prompt+2}. {self.llm_engine.prompt_data['io_structure']['input_keys']}:"
        )
        return split_text[0] if len(split_text) > 1 else text

    def generate_caption(self, prompt_handler: PromptingHandler, split_name="val"):
        output_fname = self.get_output_file_name(split_name)
        if os.path.exists(output_fname):
            logger.info(f"Caption data already exists. You can load it from cache {output_fname}")
            return

        self.model, self.processor, self.tokenizer = load_model_and_processors(
            self.args.gen_model_name, self.args.device, self.args.autocast_dtype
        )
        collate_fn = collate_fn_builder(self.processor, self.tokenizer)
        dataloader = self._initialize_dataloader(split_name, prompt_handler, collate_fn)

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )

        logger.info(f" Num examples: \t{len(dataloader.dataset)}")
        logger.info(f" Batch size: \t{dataloader.batch_size}")
        logger.info(f" Num iterations: \t{len(dataloader.dataset) // dataloader.batch_size}")

        all_generated_captions = []
        question_ids = []
        questions = []
        answers = []
        for batch in tqdm(dataloader, desc="Generating captions"):
            generated_captions = self._generate(batch)
            all_generated_captions.extend(generated_captions)
            question_ids.extend(batch["question_id"])
            questions.extend(batch["answer"])

        self.free_memory("model")  # free memory after caption generation

        all_generated_captions = self.apply_formatting(all_generated_captions, prompt_handler)
        generated_captions = self.dense_captioning_using_llm(all_generated_captions, questions)

        output = {}
        for question_id, caption in zip(question_ids, generated_captions):
            output[question_id] = caption

        for question, caption, answer in zip(questions, generated_captions, answers):
            logger.info(f"Question = {question}, Caption = {caption}, Answer = {answer}")

        self.save_to_json(output, output_fname)

    def apply_formatting(self, generated_captions: List[str], prompt_handler) -> List[str]:
        # these are fixed streps for all caption generation
        if prompt_handler.prompt_name == "prefix_a_photo_of":
            generated_captions = [f"A photo of {gen_caption}" for gen_caption in generated_captions]
        elif not prompt_handler.prompt_name.startswith("prefix_"):
            generated_captions = [
                prompt_handler.generic_prompting_handler_for_vqa({"caption": ex}) for ex in generated_captions
            ]
        generated_captions = [caption.replace("\n", "").strip() for caption in generated_captions]
        generated_captions = [re.sub(r"^\s+|\s+$", "", caption) for caption in generated_captions]
        generated_captions = [o + "." if not o.endswith(".") else o for o in generated_captions]

        return generated_captions

    def generate_dense_caption_from_messages(
        self,
        message_groups: List[List[str]],
        max_gpu_batch: int = 16,
        max_length: int = 128,
        num_beams=3,
        num_examples_in_task_prompt: int = 8,
    ) -> List[str]:
        all_generated_outputs = []

        for i in tqdm(range(0, len(message_groups), max_gpu_batch), desc="Generating dense captions"):
            message_group = message_groups[i : i + max_gpu_batch]
            chunked_input_text = [
                " ".join([message.capitalize() + ("." if not message.endswith(".") else "") for message in group])
                for group in message_group
            ]
            all_prompts = [
                self.llm_engine._prepare_prompt(text, num_examples_in_task_prompt) for text in chunked_input_text
            ]
            logger.debug(f"all prompts: {json.dumps(all_prompts, indent=2)}")

            if self.llm_engine.vllm_enabled:
                from vllm import LLM, SamplingParams

                sampling_params = SamplingParams(
                    n=1,
                    best_of=num_beams,
                    max_tokens=max_length,
                )
                generated_outputs = self.llm_engine.llm.generate(
                    prompts=all_prompts,
                    sampling_params=sampling_params,
                    use_tqdm=False,
                )
                generated_outputs = [output.outputs[0].text for output in generated_outputs]

            else:
                device = self.llm_engine.llm.device
                encodings = self.llm_engine.tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True)
                input_ids = encodings.input_ids.to(device)
                attention_mask = encodings.attention_mask.to(device)

                generated_outputs = self.llm_engine.generate_output_batch(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=max_length,
                    num_beams=num_beams,
                )

            generated_outputs = [
                self.extract_desired_text(text, num_examples_in_task_prompt).replace("\n", "").strip()
                for text in generated_outputs
            ]
            generated_outputs = [re.sub(r"^\s+|\s+$", "", caption) for caption in generated_outputs]
            all_generated_outputs.extend(generated_outputs)

            logger.debug(f"generated outputs: {json.dumps(generated_outputs, indent=2)}")
            for msg_group, dense_caption in zip(message_group[:-5], generated_outputs[:-5]):
                logger.debug(f"Message group: {msg_group}, Dense caption: {dense_caption}")

        return all_generated_outputs

    def dense_captioning_using_llm(self, generated_captions: List[str], questions: List[str]):
        if len(generated_captions) == len(questions):
            return generated_captions

        self.llm_engine = self._load_llm_engine()
        n_seq = len(generated_captions) // len(questions)
        message_groups = [generated_captions[i : i + n_seq] for i in range(0, len(generated_captions), n_seq)]

        max_gpu_batch_size = 16 * self.gpu_count
        dense_captions = self.generate_dense_caption_from_messages(
            message_groups, max_gpu_batch=max_gpu_batch_size, num_examples_in_task_prompt=4
        )

        return dense_captions

    # temperature=0.8, top_p=0.95
    def generate_blip2(
        self, inputs, max_new_tokens=128, do_sample=True, temperature=0.95, top_p=0.6, num_return_sequences=5
    ):
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                num_return_sequences=num_return_sequences,
            )
        generated_outputs = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_outputs

    def generate_kosmos(
        self, inputs, max_new_tokens=128, do_sample=True, temperature=0.95, top_p=0.6, num_return_sequences=1
    ):
        generated_ids = self.model.generate(
            pixel_values=inputs["pixel_values"].to("cuda"),
            input_ids=inputs["input_ids"][:, :-1].to("cuda"),
            attention_mask=inputs["attention_mask"][:, :-1].to("cuda"),
            img_features=None,
            img_attn_mask=inputs["img_attn_mask"][:, :-1].to("cuda"),
            use_cache=True,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            num_return_sequences=num_return_sequences,
        )
        generated_ids = generated_ids[:, inputs["input_ids"].shape[1] - 1 :]
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        generated_outputs, entities_list = zip(*[self.processor.post_process_generation(gt) for gt in generated_texts])
        entity_lists_str = [", ".join(item[0] for item in sublist) for sublist in entities_list]
        generated_outputs = [
            f"{text.rstrip('.')} ({entities})" for text, entities in zip(generated_outputs, entity_lists_str)
        ]
        logger.debug(f"GROUNDING: {generated_outputs}")

        return generated_outputs

    def generate_hfformer(
        self,
        inputs,
        max_new_tokens: int = 50,
        num_beams: int = 5,
        length_penalty: float = 1.4,
        no_repeat_ngram_size: int = 3,
    ):
        generated_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=num_beams,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        output = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

        return output

    def generate_flamingo(self, inputs, max_length=64, num_return_sequences=3):
        batch_images = inputs["image_tensors"].to(self.device)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)
        with torch.cuda.amp.autocast(dtype=self.args.autocast_dtype):
            generated_ids = self.model.generate(
                vision_x=batch_images,
                lang_x=input_ids,
                attention_mask=attention_mask,
                min_new_tokens=0,
                max_new_tokens=max_length,
                num_beams=5,
                num_return_sequences=num_return_sequences,
            )
            generated_ids = generated_ids[:, inputs["input_ids"].shape[1] :]
            generated_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            generated_outputs = [re.split(r"[\n.]", o)[0] for o in generated_outputs]

        return generated_outputs

    def generate_llava(self, batch, max_length=128, do_sample=True, num_return_sequences=3):
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
        generated_outputs = [out.strip() for out in generated_outputs]
        generated_outputs = [out[: -len(stop_str)] if out.endswith(stop_str) else out for out in generated_outputs]

        return generated_outputs

    def _generate(self, batch):
        if self.args.gen_model_name.startswith("blip"):
            inputs = {
                "input_ids": batch["input_ids"].to(self.model.device),
                "attention_mask": batch["attention_mask"].to(self.model.device),
                "pixel_values": batch["pixel_values"].to(self.model.device, self.args.autocast_dtype),
            }
            generated_captions = self.generate_blip2(inputs)

        elif self.args.gen_model_name.startswith("kosmos"):
            generated_captions = self.generate_kosmos(batch)

        elif self.args.gen_model_name.startswith("open_flamingo"):
            generated_captions = self.generate_flamingo(batch)

        elif self.args.gen_model_name == "llava":
            generated_captions = self.generate_llava(batch)

        else:
            inputs = {
                "input_ids": batch["input_ids"].to(self.device),
                "attention_mask": batch["attention_mask"].to(self.device),
                "pixel_values": batch["pixel_values"].to(self.device, self.args.autocast_dtype),
            }
            generated_captions = self.generate_hfformer(
                **inputs, max_new_tokens=50, num_beams=3, length_penalty=1.4, no_repeat_ngram_size=3
            )

        logger.debug(f"Generated captions: {json.dumps(generated_captions, indent=2)}")

        return generated_captions

    def save_to_json(self, output_captions, fname: str):
        with open(fname, "w") as f:
            json.dump(output_captions, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved generated captions to {fname}")

    def get_output_dir_path(self, split_name) -> str:
        required_args = [self.args.dataset_name, self.args.gen_model_name, self.args.vqa_format, self.args.prompt_name]
        if any(val is None for val in required_args):
            raise ValueError(
                f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
                f"Provided: {required_args}"
            )
        subdir_prefix = self.args.prompt_name.split(",")[1]
        # TODO: design a better way to handle this
        path_components = [
            OUTPUT_DIR,
            "cache",
            "generated_caption_dumps",
            self.args.dataset_name,
            self.args.gen_model_name,  # "git_large_textcaps",
            self.args.vqa_format,
            subdir_prefix,
            split_name,
        ]
        if self.args.chunk_id is not None:
            path_components.append("chunked")
            path_components.append(f"chunk{self.args.chunk_id}")

        dir_path = os.path.join(*path_components)
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

    def get_prompt_str_for_prefix_caption(self, prompt_handler: PromptingHandler, questions: List[str]) -> List[str]:
        if "promptcap" in prompt_handler.prompt_name:
            prompt_txt = []
            for q in questions:
                prompt_txt.append(prompt_handler.prompt.apply({"question": q})[0])
        elif prompt_handler.prompt_name.startswith("prefix_"):
            prompt_txt = prompt_handler.prompt.apply({})[0]
            if self.args.gen_model_name == "kosmos2":
                prompt_txt = f"<grounding> {prompt_txt}"
            logger.debug(f"PROMPT FOR CAPTION GENERATION: {prompt_txt}")
        else:
            prompt_txt = ""

        if isinstance(prompt_txt, str):
            prompt_txt = [prompt_txt] * len(questions)

        return prompt_txt

    def free_memory(self, model_name):
        del self.__dict__[model_name]
        torch.cuda.empty_cache()


class CaptionGeneratorWinoground(CaptionGenerator):
    NUM_IMAGES = 2

    def batchify_winoground(self, winoground_example):
        questions = [winoground_example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
        images = [winoground_example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
        batch = {"questions": questions, "images": images}
        batch["question_ids"] = winoground_example["id"]
        return batch

    def generate_caption(self, prompt_handler: PromptingHandler):
        output_fname = self.get_output_file_name(split_name="test")
        if os.path.exists(output_fname):
            logger.info(f"Caption data already exists. You can load it from cache {output_fname}")
            return

        dataloader = load_dataset("facebook/winoground", use_auth_token=True)["test"]
        self.model, self.processor, _ = load_model_and_processors(
            self.args.gen_model_name, self.args.device, self.args.autocast_dtype
        )

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        results = {}
        all_generated_captions = []
        questions = []
        for example in tqdm(dataloader):
            caption_texts = [example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
            images = [example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
            prompts = self.get_prompt_str_for_prefix_caption(prompt_handler, caption_texts)  # prefix for caption

            encodings = self.processor(
                text=prompts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            generated_captions = self._generate(encodings)

            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                caption_example = {f"caption_{i}": generated_captions[i] for i in range(self.NUM_IMAGES)}
                generated_captions = apply_prompt_to_example_winoground(prompt_handler, caption_example)

            all_generated_captions.extend(generated_captions)
            questions.extend(caption_texts)

        generated_captions = self.dense_captioning_using_llm(all_generated_captions, questions)

        # assign two captions to each example
        logger.debug(f"all_generated_captions len: {len(generated_captions)}")
        for i, example in enumerate(dataloader):
            results[example["id"]] = [generated_captions[i * 2], generated_captions[i * 2 + 1]]

        self.save_to_json(results, output_fname)


class PromptCapGenerarator(CaptionGenerator):
    def __init__(self, args, device="cuda"):
        self.args = args
        self.device = device

    def _load_model(self, device):
        self.promptcap = PromptCap("vqascore/promptcap-coco-vqa")  # Load the PromptCap model
        self.promptcap.model.to(device)

    def _initialize_dataloader(self, split_name: str):
        config = copy.copy(self.args)
        if "train" in split_name:  # hack to avoid chunking for train split for few-shot inference (cvpr submission)
            config.chunk_id = None

        batch_size = 16  # hardcoded for now
        dataset_args = {
            "config": config,
            "dataset_name": self.args.dataset_name,
            "split_name": split_name,
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

        return dataloader

    def generate_caption(self, prompt_handler: PromptingHandler, split_name: str):
        output_fname = self.get_output_file_name(split_name)
        if os.path.exists(output_fname):
            logger.info(f"Caption data already exists. You can load it from cache {output_fname}")
            return

        self._load_model(self.device)

        dataloader = self._initialize_dataloader(split_name)

        logger.info(
            f" Generating caption data for {self.args.dataset_name} dataset, {split_name} split."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )
        logger.info(f" Num examples: \t{len(dataloader.dataset)}")
        logger.info(f" Batch size: \t{dataloader.batch_size}")
        logger.info(f" Num iterations: \t{len(dataloader.dataset) // dataloader.batch_size}")

        results, cached_ids = self.load_data_from_cache(split_name)
        for batch in tqdm(dataloader, desc="Generating captions"):
            question_ids = batch["question_id"]
            images = batch["image"]
            questions = batch["question"]
            # check if all the question_ids are already in cached_ids using all() function
            if all([qid in cached_ids for qid in question_ids]):
                continue

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

            for question_id, caption in zip(question_ids, generated_captions):
                results[question_id] = caption

            if len(results) % 100 == 0:
                cache_fname = self.get_cache_file_path(split_name)
                self.save_to_json(results, cache_fname)

        self.save_to_json(results, output_fname)

    def generate_standard(self, images: List[Image.Image], questions: List[str]):
        generated_captions = []
        for image, question in zip(images, questions):
            prompt = f"Please describe this image according to the given question: {question}"
            output = self.promptcap.caption(prompt, image)
            generated_captions.append(output)
        return generated_captions

    def load_data_from_cache(self, data_split: str):
        cached_data = {}
        cache_file_path = self.get_cache_file_path(data_split)
        if os.path.exists(cache_file_path):
            with open(cache_file_path, "r") as file:
                cached_data = json.load(file)
            logger.info(f"Loaded cached data from {cache_file_path}")
        cached_ids = set(cached_data.keys())
        return cached_data, cached_ids


class PromptCapGeneratorWinoground(PromptCapGenerarator):
    NUM_IMAGES = 2

    def batchify_winoground(self, winoground_example):
        questions = [winoground_example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
        images = [winoground_example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
        batch = {"questions": questions, "images": images}
        batch["question_ids"] = winoground_example["id"]
        return batch

    def generate_caption(self, prompt_handler: PromptingHandler):
        output_fname = self.get_output_file_name(split_name="test")
        if os.path.exists(output_fname):
            logger.info(f"Caption data already exists. You can load it from cache {output_fname}")
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

        self.save_to_json(output, output_fname)

    def generate_standard(self, images: List[Image.Image], questions: List[str]):
        generated_captions = []
        for image, question in zip(images, questions):
            prompt = f"Please describe this image according to the given question: {question}"
            output = self.promptcap.caption(prompt, image)
            generated_captions.append(output)
        return generated_captions
