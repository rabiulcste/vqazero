import json
import os
import re
from typing import List, Union

import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from evals.vqa_accuracy import eval_winoground
from utils.config import OUTPUT_DIR
from utils.handler import PromptingHandler
from utils.logger import Logger
from vqa_zero.caption_generation import (CaptionGeneratorWinoground,
                                         PromptCapGeneratorWinoground)
from vqa_zero.inference_utils import (get_output_dir_path,
                                      get_prompt_template_handler,
                                      load_model_and_processors, save_to_json)

logger = Logger(__name__)

NUM_IMAGES = 2

# winoground-dataset
winoground = load_dataset("facebook/winoground", use_auth_token=True)["test"]


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

    def apply_prompt_to_example(self, handler: PromptingHandler, example):
        return (
            handler.gpt3_baseline_qa_prompting_handler_for_winoground(example)
            if handler.prompt_name.startswith("prefix_")  # means it's a cached question prompt
            else handler.generic_prompting_handler_for_winoground(example)
        )

    def get_cached_context_file_path(self, split_name: str) -> str:
        required_args = [
            self.args.dataset_name,
            self.args.gen_model_name,
            self.args.vqa_format,
            self.args.prompt_name,
        ]
        if any(val is None for val in required_args):
            raise ValueError(
                f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
                f"Provided: {required_args}"
            )
        subdir_prefix = self.args.prompt_name.split(",")[1]
        context_dir_str = (
            "generated_caption_dumps" if self.args.vqa_format == "caption_vqa" else "generated_rationale_dumps"
        )
        dir_path = os.path.join(
            OUTPUT_DIR,
            "cache",
            context_dir_str,
            self.args.dataset_name,
            self.args.gen_model_name,
            self.args.vqa_format,
            subdir_prefix,
            split_name,
        )
        os.makedirs(dir_path, exist_ok=True)
        output_file_path = os.path.join(dir_path, f"output.json")
        return output_file_path

    def load_cached_context_data(self, split_name: str):
        context_data_file_path = self.get_cached_context_file_path(split_name)
        if not os.path.exists(context_data_file_path):
            raise FileNotFoundError(
                f"File {context_data_file_path} does not exist. Please generate the prompted captions first.\n"
                f"python caption_generation.py --dataset_name {self.config.dataset_name} "
                f"--model_name {self.config.model_name} --vqa_format caption_qa "
                f"--prompt_name {self.config.prompt_name}"
            )
        with open(context_data_file_path, "r") as f:
            prompted_captions = json.load(f)
        logger.info(f"Loaded prompted captions from {context_data_file_path}")
        cached_data = prompted_captions

        return cached_data

    def _get_captions_by_eid(self, eid):
        if self.args.vqa_format == "standard_vqa":
            return []

        return self.additional_context_data[str(eid)]

    def get_formatted_captions(self, captions: List[str]):
        return [f"Visual Context: {caption}" for caption in captions]

    def _perform_vqa_inference(self):
        device = self.get_device()
        autocast_dtype = self.get_autocast_dtype()
        self.args.autocast_dtype = autocast_dtype

        logger.info(f'Selected prompt name :"{self.args.prompt_name}"')
        prompt_handler, template_expr = get_prompt_template_handler(self.args, self.args.prompt_name)

        if isinstance(prompt_handler, List):
            prompt_handler, context_prompt_handler = prompt_handler
            self._handle_caption_and_cot_generation(context_prompt_handler)
            self.additional_context_data = self.load_cached_context_data("test")

        model, processor, tokenizer = load_model_and_processors(self.args.model_name, device, self.args.autocast_dtype)

        predictions = []
        for eid, example in enumerate(tqdm(winoground)):
            images = [example[f"image_{i}"].convert("RGB") for i in range(NUM_IMAGES)]
            questions = self.apply_prompt_to_example(prompt_handler, example)
            model_generated_captions = self._get_captions_by_eid(eid)
            model_generated_captions = self.get_formatted_captions(model_generated_captions)
            if model_generated_captions:
                prompt = []
                for q in questions:
                    for c in model_generated_captions:
                        prompt.append(f"{c}\n{q}")  # [c0, c0, c1, c1]
            else:
                prompt = [q for q in questions for _ in range(2)]  # [c0, c0, c1, c1]

            image = [images[i % 2] for i in range(4)]  # [i0, i1, i0, i1]

            if example["id"] < 3:
                message = "\n".join(questions)
                logger.debug(f"Example Prompt Questions = {message}")

            if self.args.model_name == "kosmos2":
                inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to(model.device)
                generated_ids = model.generate(
                    pixel_values=inputs["pixel_values"].to("cuda"),
                    input_ids=inputs["input_ids"][:, :-1].to("cuda"),
                    attention_mask=inputs["attention_mask"][:, :-1].to("cuda"),
                    img_features=None,
                    img_attn_mask=inputs["img_attn_mask"][:, :-1].to("cuda"),
                    max_new_tokens=self.args.max_length,
                    length_penalty=self.args.length_penalty,
                    num_beams=self.args.num_beams,
                )
                generated_ids = generated_ids[:, inputs["input_ids"].shape[1] - 1 :]
                generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
                generated_outputs, entities_list = zip(
                    *[processor.post_process_generation(gt) for gt in generated_texts]
                )

            elif "open_flamingo" in self.args.model_name:
                b_images = [[im] for im in image]
                batch_images = processor._prepare_images(b_images).to(device)
                input_ids, attention_mask = processor._prepare_text(prompt)

                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)

                with torch.cuda.amp.autocast(dtype=autocast_dtype):
                    generated_ids = model.generate(
                        vision_x=batch_images,
                        lang_x=input_ids,
                        attention_mask=attention_mask,
                        min_new_tokens=0,
                        max_new_tokens=self.args.max_length,
                        num_beams=3,
                        length_penalty=self.args.length_penalty,
                    )

                generated_ids = generated_ids[:, input_ids.shape[1] - 1 :]
                generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            elif self.args.model_name == "llava":
                aggregate_output = []
                for txt, img in zip(prompt, image):
                    image_tensor, input_ids = processor.get_processed_tokens(txt, img)
                    input_ids = input_ids.to("cuda", non_blocking=True)

                    with torch.inference_mode():
                        generated_ids = model.generate(
                            input_ids,
                            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
                            num_beams=self.args.num_beams,
                            max_new_tokens=self.args.max_length,
                            length_penalty=self.args.length_penalty,
                            use_cache=True,
                        )
                    generated_ids = generated_ids[:, input_ids.shape[1] :]
                    generated_outputs = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

                    aggregate_output.append(generated_outputs[0])
                generated_outputs = aggregate_output

            else:
                inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to(model.device)
                generated_ids = model.generate(**inputs)
                generated_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

            generated_outputs = [out.strip() for out in generated_outputs]
            generated_outputs = self.yes_no_parser(generated_outputs)
            logger.debug(f"Output: {generated_outputs}")
            scores = {
                "c0_i0": generated_outputs[0],
                "c0_i1": generated_outputs[1],
                "c1_i0": generated_outputs[2],
                "c1_i1": generated_outputs[3],
            }
            curr_score = {
                "id": example["id"],
                "scores": scores,
                "prompt": prompt,
            }
            logger.debug(f"Example {eid} score: {curr_score}")
            predictions.append(curr_score)
        return predictions

    @staticmethod
    def yes_no_parser(result: List[str]):
        outputs = []
        for sentence in result:
            sentence = sentence.strip('"\n ')
            # Use a case-insensitive regex pattern to capture "yes" or "no" at the beginning of the sentence
            match = re.search(r"\b(?i)(yes|no)\b", sentence)
            if match:
                outputs.append(match.group().lower())
            else:
                outputs.append(sentence)  # Or you can choose another default value
        return outputs

    def _handle_caption_and_cot_generation(self, prompt_handler: PromptingHandler) -> None:
        # Implementation of the caption and chain-of-thought generation logic
        if self.args.gen_model_name is None:
            self.args.gen_model_name = self.args.model_name

        def get_caption_generator(args):
            if args.gen_model_name == "promptcap":
                return PromptCapGeneratorWinoground(args, device="cuda")
            else:
                return CaptionGeneratorWinoground(args, device="cuda")

        if self.args.vqa_format == "caption_vqa":
            caption_gen = get_caption_generator(self.args)
            if self.args.few_shot:
                caption_gen.generate_caption(prompt_handler)
            caption_gen.generate_caption(prompt_handler)


def main(args):
    """
    Performs Visual Question Answering (VQA) inference on the Winoground dataset using the huggingface transformer models.
    The function supports three different formats for the VQA task:
        - standard_vqa: the model is given an image and a question and it has to answer the question.
        - caption_qa: the model is given an image, a question, and a caption. The model has to answer the question,
          with the caption providing context for improving the answer.

    Notes:
        - The function loads the transformer model and any necessary pre-processors based on the arguments.
        - If the `vqa_format` argument is set to "caption_qa" and the `model_name` argument is set to "blip_vqa",
          the function loads a separate caption model and pre-processors.
        - If the `vqa_format` argument is set to "decompose_qa", the function calls `call_lavis_blip_iterative`,
          which doesn't return scores.
    """
    output_dir = get_output_dir_path(args)
    if os.path.exists(output_dir) and not args.overwrite_output_dir:
        logger.info(f"Output directory {output_dir} already exists. Skipping inference.")
        return

    # get prompting handler
    if args.prompt_name is None:
        raise ValueError("Prompt name must be provided.")

    args.num_beams = 3
    args.max_length = 5
    args.length_penalty = 1.0

    inf = VQAInference(args)
    predictions = inf._perform_vqa_inference()

    # Save the results as a JSON file
    prediction_file_path = os.path.join(output_dir, "predictions.json")
    save_to_json(prediction_file_path, predictions)
    eval_winoground(args, output_dir, predictions)
