import os
from typing import List, Union

from string import punctuation
from datasets import load_dataset
# pip install accelerate
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch
from modeling_utils import (get_dir_path, get_prompt_handler,
                            load_model_and_processors)
from utils.globals import MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.logger import Logger
from utils.vqa_accuracy import eval_winoground

logger = Logger(__name__)


# TODO: integrate nl-augmenter
NUM_IMAGES = 2

# winoground-dataset
winoground = load_dataset("facebook/winoground", use_auth_token=True)["test"]


# TODO: Need to test it
"""
To use it in transformers, please refer to https://github.com/OFA-Sys/OFA/tree/feature/add_transformers. Install the transformers and download the models as shown below.

git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
git clone https://huggingface.co/OFA-Sys/OFA-huge-vqa
"""

RESOLUTION = 480
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]

patch_resize_transform = transforms.Compose(
    [
        lambda image: image.convert("RGB"),
        transforms.Resize((RESOLUTION, RESOLUTION), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ]
)


def ofa_vqa(args, model, processor):
    from generate import sequence_generator
    from torchvision import transforms
    from transformers import OFAModel, OFATokenizer

    txt = ""  # some text string
    inputs = processor([txt], return_tensors="pt").input_ids
    img = ""  # some PIL image
    patch_img = patch_resize_transform(img).unsqueeze(0)

    generated_tokens = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3)
    output = processor.batch_decode(generated_tokens, skip_special_tokens=True)
    logger.info(f"OFA-VQA Output: {output}")


def apply_prompt_to_example(handler: PromptingHandler, example):
    return (
        handler.gpt3_baseline_qa_prompting_handler_for_winoground(example)
        if handler.prompt_name.startswith("prefix_")  # means it's a cached question prompt
        else handler.generic_prompting_handler_for_winoground(example)
    )


def run_standard_qa(args, model, processor, prompt_handler: PromptingHandler):
    predictions = []
    for eid, example in enumerate(tqdm(winoground)):
        images = [example[f"image_{i}"].convert("RGB") for i in range(NUM_IMAGES)]
        questions = apply_prompt_to_example(prompt_handler, example)

        if example["id"] < 3:
            logger.info("Example Prompt Questions:")
            message = "\n".join(questions)
            logger.info(message)

        image = [images[i % 2] for i in range(4)]  # [i0, i1, i0, i1]
        prompt = [q for q in questions for _ in range(2)]  # [c0, c0, c1, c1]

        inputs = processor(images=image, text=prompt, padding=True, return_tensors="pt").to(model.device)
        generated_ids = model.generate(**inputs)
        output = processor.batch_decode(generated_ids, skip_special_tokens=True)

        logger.debug(f"Output: {output}")
        curr_score = {
            "id": example["id"],
            "c0_i0": output[0],
            "c0_i1": output[1],
            "c1_i0": output[2],
            "c1_i1": output[3],
        }
        predictions.append(curr_score)
    return predictions


def run_caption_qa(
    args,
    model,
    processor,
    prompt_handler: Union[PromptingHandler, List[PromptingHandler]],
):
    if isinstance(prompt_handler, List):
        handler_1, handler_2 = prompt_handler
    else:
        raise ValueError("handler should be a list of two handlers")

    predictions = []
    for eid, example in enumerate(tqdm(winoground)):
        images = [example[f"image_{i}"].convert("RGB") for i in range(NUM_IMAGES)]
        samples = processor(images=images, return_tensors="pt").to(model.device)
        generated_ids = model.generate(
            **samples,
            repetition_penalty=1.5,
        )
        model_generated_captions = processor.batch_decode(generated_ids, skip_special_tokens=True)
        model_generated_captions = [caption.strip() for caption in model_generated_captions]

        caption_example = {
            "caption_0": model_generated_captions[0],
            "caption_1": model_generated_captions[1],
        }
        model_generated_captions = handler_2.generic_prompting_handler(caption_example)

        questions = apply_prompt_to_example(handler_1, example)

        if example["id"] < 1:
            logger.info("Example Prompt Questions:")
            questions = "\n".join(questions)
            logger.info(questions)

            logger.info(f"Generated captions:")
            model_generated_captions = "\n".join(model_generated_captions)
            logger.info(model_generated_captions)

        images = [images[i % 2] for i in range(4)]  # [i0, i1, i0, i1]

        prompt = []
        for q in questions:
            for c in model_generated_captions:
                prompt.append(f"{c}\n{q}")  # [c0, c0, c1, c1]

        logger.info(f"Input Prompt: {prompt}")
        inputs = processor(images=images, text=prompt, padding=True, return_tensors="pt").to(model.device)

        output = model.generate(**inputs, repetition_penalty=1.5)

        # Group the scores by caption and image and return them
        curr_score = {
            "id": example["id"],
            "c0_i0": output[0],
            "c0_i1": output[1],
            "c1_i0": output[2],
            "c1_i1": output[3],
        }
        predictions.append(curr_score)
    return predictions


def run_inference(args):
    """
    Performs Visual Question Answering (VQA) inference on the Winoground dataset using the huggingface transformer models.
    The function supports three different formats for the VQA task:
        - basic_qa: the model is given an image and a question and it has to answer the question.
        - caption_qa: the model is given an image, a question, and a caption. The model has to answer the question,
          with the caption providing context for improving the answer.
        - decompose_qa: the model is given an image, a question, and a decomposition of the question. The model has to
          answer the question, with the decomposition providing context for improving the answer.

    Notes:
        - The function loads the transformer model and any necessary pre-processors based on the arguments.
        - If the `vqa_format` argument is set to "caption_qa" and the `model_name` argument is set to "blip_vqa",
          the function loads a separate caption model and pre-processors.
        - If the `vqa_format` argument is set to "decompose_qa", the function calls `call_lavis_blip_iterative`,
          which doesn't return scores.
        - For all formats except "decompose_qa", the function calls `compute_and_save_winoground_scores` to
          compute and save the scores.
    """

    device = args.device
    model, processor = load_model_and_processors(args.model_name, device)
    output_dir = get_dir_path(args)
    if os.path.exists(output_dir) and not args.overwrite_output_dir:
        logger.info(f"Output directory {output_dir} already exists. Skipping inference.")
        return

    # get prompting handler
    prompt_name = None
    if args.prompt_name:
        prompt_name = args.prompt_name
        if len(args.prompt_name.split(",")) > 1:
            prompt_name = args.prompt_name.split(",")

    logger.info(f'Selected prompt name :"{prompt_name}"')
    handler, template_expr = get_prompt_handler(args, prompt_name)
    args.num_beams = 5
    args.max_length = 10
    print(processor.tokenizer.cls_token_id)
    if args.vqa_format == "basic_qa":
        predictions = run_standard_qa(args, model, processor, handler)
    elif args.vqa_format == "caption_qa":
        predictions = run_caption_qa(args, model, processor, handler)
    else:
        raise NotImplementedError(
            f"Provided VQA format {args.vqa_format} is either not implemented yet or invalid argument provided."
        )
    eval_winoground(args, output_dir, predictions, template_expr)
