import json
import os
import time
import time
from string import punctuation
from typing import List, Union
from typing import List, Union

import spacy
import torch
from promptcap import PromptCap
from torch.utils.data import DataLoader
from promptcap import PromptCap
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, Blip2ForConditionalGeneration

from dataset_zoo.custom_dataset import VQADataset, collate_fn
from dataset_zoo.custom_dataset import VQADataset, collate_fn
from modeling_utils import (apply_prompt_to_example_winoground,
                            get_image_tensors_winoground,
                            get_optimal_batch_size, get_optimal_batch_size_v2,
                            load_model_and_processors)
from utils.config import PROJECT_DIR
from utils.globals import MODEL_CLS_INFO
                            get_image_tensors_winoground,
                            get_optimal_batch_size, get_optimal_batch_size_v2,
                            load_model_and_processors)
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from utils.globals import MODEL_CLS_INFO
from utils.config import PROJECT_DIR
from utils.handler import PromptingHandler
from utils.logger import Logger

logger = Logger(__name__)



nlp = spacy.load("en_core_web_sm")

# This class exclusively loads cached caption data, primarily designed for use with the
# inference_vqa.py script. The script assumes that captions have already been generated and saved
# in a file. To generate captions, please refer to the caption_generator.py and pipeliner.py scripts.
N = 5  # number of days
split_map = {
    "train": "train",
    "val": "",
    "testdev": "testdev",
}

class CaptionPrefixHandler:
    def __init__(self, args, device="cuda"):
    def __init__(self, args, device="cuda"):
        self.args = args
        self.data = None  # caption data
        self.device = device

    def generate_caption_to_use_as_prompt_prefix(self, prompt_handler: PromptingHandler, split="val"):
        self.args.split = split
        if os.path.exists(self.get_file_path()):
            file_mod_time = os.path.getmtime(self.get_file_path())
            current_time = time.time()
            two_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < two_days_in_seconds:
                logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path()}")
                return
    def generate_caption_to_use_as_prompt_prefix(self, prompt_handler: PromptingHandler, split="val"):
        # if not self.args.overwrite_output_dir and os.path.exists(self.get_file_path(split)):
        if os.path.exists(self.get_file_path(split)):
            file_mod_time = os.path.getmtime(self.get_file_path(split))
            current_time = time.time()
            two_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < two_days_in_seconds:
                logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path(split)}")
                return

        model_name = MODEL_CLS_INFO["hfformer"][self.args.gen_model_name]["name"]
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="auto"
        )

        batch_size = get_optimal_batch_size(self.args)
        dataset_args = {
            "config": self.args,
            "dataset_name": self.args.dataset_name,
            # "processor": self.processor,
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
            f" Generating caption data for {self.args.dataset_name} dataset."
            f" It may take a few minutes depending on your dataset size. Please wait..."
        )

        logger.info(f" Num examples: \t{len(dataset)}")
        logger.info(f" Batch size: \t{batch_size}")
        logger.info(f" Num iterations: \t{len(dataset) // batch_size}")

        output = {}
        for batch in tqdm((dataloader), desc="Generating captions"):
            images = batch["images"]
            questions = batch["questions"]
            answers = batch["answers"]
            prompt = self.get_prompt_str_for_prefix_caption(prompt_handler, questions)  # prefix for caption
            # print(prompt)
            inputs = self.processor(images=images, text=prompt, padding=True, return_tensors="pt").to(
                self.device, torch.bfloat16
            )
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                # num_beam_groups=5,
                # do_sample=True,
                num_return_sequences=1,
                # top_p=0.9,
                # top_k=50,
                # temperature=0.7,
                # length_penalty=2.0,
                # no_repeat_ngram_size=3,
            )
            generated_captions = self.processor.batch_decode(generated_ids, skip_special_tokens=True)

            if len(generated_captions) != len(questions):
                new_list = []
                for i in range(0, len(generated_captions), 3):
                    current_caption_list = generated_captions[i : i + 3]
                    print(current_caption_list)
                    # take the maximum length caption
                    current_caption = max(current_caption_list, key=len)
                    new_list.append(current_caption)
                generated_captions = new_list

            # these are fixed streps for all caption generation
            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
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
                output[question_id] = caption.strip().replace("\n", "")

            for question, caption, answer in zip(questions, generated_captions, answers):
                print(f"Question: {question} | Caption: {caption} | Answer: {answer}")

        self.save(output, split)

    def generate_blip(self, samples):
        with torch.no_grad(), torch.cuda.amp.autocast():
            if self.args.model_name == "blip_vqa":
                return self.model.generate(samples=samples)
            else:
                return self.model.generate(
                    samples=samples, max_length=50, length_penalty=1.45, num_beams=5, no_repeat_ngram_size=3
                )

    def generate_hfformer(self, samples, max_new_tokens, num_beams=5, length_penalty=1.4, no_repeat_ngram_size=3):
        # print(f"samples: {samples} | max_new_tokens: {max_new_tokens} | num_beams: {num_beams} | length_penalty: {length_penalty} | no_repeat_ngram_size: {no_repeat_ngram_size}")
        # with torch.no_grad(), torch.cuda.amp.autocast():
        generated_ids = self.model.generate(
            **samples,
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

    def generate_standard(self, image_tensors, questions: List[str], prompt_handler: PromptingHandler):
        prompt_txt = self.get_prompt_str_for_prefix_caption(prompt_handler, questions)  # prefix for caption

        if self.args.model_name.startswith("blip"):
            samples = {
                "image": torch.stack(image_tensors).to(self.model.device),
                "prompt": prompt_txt,
            }
            generated_captions = self.generate_blip(samples)
        elif self.args.model_name == "open_flamingo_lamma":
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
            input_ids = self.processor(text=prompt_txt, add_special_tokens=False).input_ids
            samples = {
                "input_ids": input_ids,
                "pixel_values": torch.stack(image_tensors).to(self.model.device),
            }
            generated_captions = self.generate_hfformer(
                samples, max_new_tokens=50, num_beams=4, length_penalty=0.6, no_repeat_ngram_size=3
            )
        return generated_captions

    def load(self, split: str):
        fname = self.get_file_path(split)
        if not os.path.exists(fname):
            raise FileNotFoundError(
                f"File {fname} does not exist. Please generate the prompted captions first.\n"
                f"python caption_generator.py --dataset_name {self.args.dataset_name} "
                f"--model_name {self.args.model_name} --vqa_format caption_qa "
                f"--prompt_name {self.args.prompt_name}"
            )
        print(f"Loading prompted captions from {fname}")
        with open(fname, "r") as f:
            prompted_captions = json.load(f)
        logger.info(f"Loaded prompted captions from {fname}")
        self.data = prompted_captions

    def load_by_ids(self, ids: Union[str, List[str]]):
        if isinstance(ids, str):
            prompted_captions_batch = self.data[str(ids)]  # handling winoground case
        else:
            prompted_captions_batch = [self.data[str(idx)] for idx in ids]
        return prompted_captions_batch

    def save(self, output_captions, split):
        fname = self.get_file_path(split)
        with open(fname, "w") as f:
            json.dump(output_captions, f, ensure_ascii=False, indent=4)

        logger.info(f"Saved generated captions to {fname}")

    def get_dir_path(self, split) -> str:
        required_args = [self.args.dataset_name, self.args.gen_model_name, self.args.vqa_format, self.args.prompt_name]
        if any(val is None for val in required_args):
            raise ValueError(
                f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
                f"Provided: {required_args}"
            )
        subdir_prefix = self.args.prompt_name.split(",")[1]
        # TODO: design a better way to handle this
        split = split_map[split]
        dir_path = os.path.join(
            PROJECT_DIR,
            "output",
            "generated_caption_dumps" if "decompose" not in self.args.prompt_name else "decomposed_dumps",
            self.args.dataset_name,
            self.args.gen_model_name,  # "git_large_textcaps",
            self.args.vqa_format,
            subdir_prefix,
            split,
        )
        os.makedirs(dir_path, exist_ok=True)

        return dir_path

    def get_file_path(self, split) -> str:
        dir_path = self.get_dir_path(split)
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


class CaptionPrefixHandlerWinoground(CaptionPrefixHandler):
    NUM_IMAGES = 2

    def batchify_winoground(self, winoground_example):
        questions = [winoground_example[f"caption_{i}"] for i in range(self.NUM_IMAGES)]
        images = [winoground_example[f"image_{i}"].convert("RGB") for i in range(self.NUM_IMAGES)]
        batch = {"questions": questions, "images": images}
        batch["question_ids"] = winoground_example["id"]
        return batch

    def generate_caption_to_use_as_prompt_prefix(self, dataloader, prompt_handler: PromptingHandler):
        if os.path.exists(self.get_file_path()):
            file_mod_time = os.path.getmtime(self.get_file_path())
            current_time = time.time()
            two_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < two_days_in_seconds:
                logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path()}")
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

            # print(generated_captions)

            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                caption_example = {f"caption_{i}": generated_captions[i] for i in range(self.NUM_IMAGES)}
                generated_captions = apply_prompt_to_example_winoground(prompt_handler, caption_example)

            output[example["id"]] = generated_captions
        self.save(output, split)

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


class PromptcapPrefixHandler(CaptionPrefixHandler):
    def __init__(self, args, device="cuda"):
        model = PromptCap("vqascore/promptcap-coco-vqa")  # Load the PromptCap model
        if device == "cuda":
            model.cuda()  # Move the model to the GPU if 'cuda' is specified as the device
        self.args = args
        self.model = model

    def generate_caption_to_use_as_prompt_prefix(self, prompt_handler: PromptingHandler, split):
        if os.path.exists(self.get_file_path(split)):
            logger.info(f"Caption data already exists. You can load it from cache {self.get_file_path(split)}")
            return

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
            image_paths = batch["image_paths"]
            images = batch["images"]
            questions = batch["questions"]

            prompted_questions = [
                f"Please describe this image according to the given question: {question}" for question in questions
            ]
            generated_captions = self.model.caption_batch(prompted_questions, images)

            # these are fixed streps for all caption generation
            if prompt_handler.prompt_name.startswith("prefix_"):
                generated_captions = [f"A photo of {gen_caption}." for gen_caption in generated_captions]
            else:
                generated_captions = [
                    prompt_handler.generic_prompting_handler_for_vqa({"caption": ex}) for ex in generated_captions
                ]
            logger.debug(f"Generated captions: {json.dumps(generated_captions)}")
            assert len(generated_captions) == len(questions)

            for question_id, caption in zip(batch["question_ids"], generated_captions):
                output[question_id] = caption

        self.save(output, split)

    # def generate_standard(self, images, questions: List[str]):
    #     prompts = [f"please describe this image according to the given question: {question}" for question in questions]
    #     return self.model.caption_batch(prompts, images)

    def generate_standard(self, images, questions):
        generated_captions = []
        for image, question in zip(images, questions):
            prompt = f"Please describe this image according to the given question: {question}"
            output = self.model.caption(prompt, image)
            generated_captions.append(output)
        return generated_captions


class BreakDownQuestion(CaptionPrefixHandler):
    def __init__(self, args, device="cuda"):
        self.args = args

    def decompse_question(self, dataloader):
        import torch
        from transformers import AutoTokenizer, T5ForConditionalGeneration

        model = T5ForConditionalGeneration.from_pretrained(
            "google/flan-ul2", torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained("google/flan-ul2")

        output = {}
        for batch in tqdm((dataloader), desc="Generating decomposed steps"):
            prompted_questions = []
            for question in batch["questions"]:
                curr_prompted_question = (
                    "To answer this question in an image, what are the steps? You can assume that one will always look at the image first. "
                    "Question: What breed this dog is? "
                    "Step1: Identify dog's characteristics. Step2: Check how large or small they are. Step3: Identify the type. "
                    "Question: What could this gentleman be carrying in that red bag? "
                    "Step1: Look at the bag. Step2: Look at the gentleman. Step3: Make a guess. "
                    f"Question: {question} "
                    "Now, show me the steps to answer this question (maximum 3)."
                )
                prompted_questions.append(curr_prompted_question)

            # print(question)
            inputs = tokenizer(prompted_questions, return_tensors="pt", padding=True).input_ids.to("cuda")
            generated_steps = model.generate(
                inputs, max_length=200, num_beams=5, repetition_penalty=2.5, early_stopping=True
            )

            for question_id, caption in zip(batch["question_ids"], generated_steps):
                output[question_id] = caption

        self.save(output)
