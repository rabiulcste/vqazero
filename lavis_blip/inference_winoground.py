import json
import os
from collections import defaultdict
from typing import List, Union

import torch
from datasets import load_dataset
from tqdm import tqdm

from caption_generation import CaptionPrefixHandlerWinoground
from modeling_utils import (apply_prompt_to_example_winoground,
                            generate_output_blip, get_dir_path,
                            get_image_tensors_winoground, get_prompt_handler,
                            load_model_and_processors, save_to_json)
from utils.handler import DecompositionHandler, PromptingHandler
from utils.logger import Logger
from utils.vqa_accuracy import eval_winoground

logger = Logger(__name__)


NUM_IMAGES = 2
# load winoground-dataset
winoground = load_dataset("facebook/winoground", use_auth_token=True)["test"]


def create_prompt_chat_str(
    args,
    vqa_responses,
    decomposed_questions,
    example_index: int,
    caption_index: int,
    decomposition_handler,
    prompt_handler,
):
    if args.decomposition_type in ["dialogue", "cot"]:
        subq_answers = [vqa_responses.get(f"c{caption_index}_i{image_index}", "") for image_index in range(NUM_IMAGES)]
        prompt_chat_str = [
            "\n".join(
                [
                    f"{q} \n {a}"
                    for q, a in zip(
                        decomposed_questions[: len(subq_answers[image_index])],
                        subq_answers[image_index],
                    )
                ]
            )
            for image_index in range(NUM_IMAGES)
        ]

        if args.decomposition_type == "cot":
            cot_prompt = decomposition_handler.get_cot_prompt(
                example_index, caption_index, decomposed_questions, prompt_handler
            )
            prompt_chat_str = [f"{cot_prompt} \n {prompt_chat_str[image_index]}" for image_index in range(NUM_IMAGES)]
    else:
        prompt_chat_str = ["" for _ in range(NUM_IMAGES)]

    return prompt_chat_str


def process_example_decompose_qa(
    args,
    example,
    model,
    vis_processors,
    txt_processors,
    decomposition_handler: DecompositionHandler,
    prompt_handler: PromptingHandler,
):
    TEXT_INPUT_KEY = "text_input" if args.model_name == "blip_vqa" else "prompt"
    image_tensors = get_image_tensors_winoground(example, vis_processors)
    sub_questions_data = decomposition_handler.get_subquestions_data(example["id"])
    logger.debug(f"SUB-QUESTIONS DATA: {sub_questions_data}")
    vqa_responses = defaultdict(list)

    for caption_index in range(NUM_IMAGES):
        decomposed_questions = sub_questions_data[f"caption_{caption_index}"]["decomposed_questions"]
        for question in decomposed_questions:
            prompt_chat_str = create_prompt_chat_str(
                args,
                vqa_responses,
                decomposed_questions,
                example["id"],
                caption_index,
                decomposition_handler,
                prompt_handler,
            )
            prompt = [prompt_chat_str[image_index] + "\n" + question for image_index in range(NUM_IMAGES)]
            logger.debug(f"PROMPT: {prompt}")

            samples = {
                "image": torch.stack([img for img in image_tensors]).to(model.device),  # [i0, i1]
                TEXT_INPUT_KEY: prompt,  # [c0, c0]
            }
            output = generate_output_blip(args, model, samples)

            for image_index in range(NUM_IMAGES):
                vqa_responses[f"c{caption_index}_i{image_index}"].append(output[image_index])

    sub_questions_data["decomposed_vqa_answers"] = vqa_responses

    return sub_questions_data


def run_decompose_qa(
    args,
    model,
    processor,
    decomposition_handler: DecompositionHandler,
):
    if isinstance(processor, tuple):
        vis_processors, txt_processors = processor
    all_results = {}

    prompt_handler = None
    if args.decomposition_type == "cot":
        prompt_handler = PromptingHandler(
            args.dataset_name, "prefix_convert_question_to_binary_vqa", subset_name="basic"
        )

    for _, example in enumerate(tqdm(winoground)):
        output = process_example_decompose_qa(
            args, example, model, vis_processors, txt_processors, decomposition_handler, prompt_handler
        )
        all_results[example["id"]] = output
        logger.info(f"DEBUG: {json.dumps(output, indent=4)}")

    json_path = os.path.join(get_dir_path(args), "output.json")
    save_to_json(json_path, all_results)


def run_vqa_inference_and_evaluation(args, model, processor, **kwargs):
    # get prompting handler
    prompt_handler, template_expr = get_prompt_handler(args)
    if isinstance(prompt_handler, List):
        prompt_handler, prompt_handler_2 = prompt_handler

    if isinstance(processor, tuple):
        vis_processors, txt_processors = processor
    TEXT_INPUT_KEY = "text_input" if args.model_name == "blip_vqa" else "prompt"

    # cache captions for caption_qa if it is not already cached
    if args.vqa_format == "caption_qa":
        capgen = CaptionPrefixHandlerWinoground(args, model, vis_processors, device="cuda")
        capgen.generate_caption_to_use_as_prompt_prefix(winoground, prompt_handler_2)
        capgen.load()

    predictions = []
    for eid, example in enumerate(tqdm(winoground)):
        image_tensors = get_image_tensors_winoground(example, vis_processors)
        questions = apply_prompt_to_example_winoground(prompt_handler, example)
        text_input = [q for q in questions for _ in range(2)]

        if args.vqa_format == "caption_qa":
            generated_captions = capgen.load_by_ids(str(example["id"]))
            text_input = []
            for q in questions:
                for c in generated_captions:
                    text_input.append(f"{c}\n{q}")
            logger.info(f"Text input: {json.dumps(text_input, indent=2)}")

        samples = {
            "image": torch.stack([image_tensors[i % 2] for i in range(4)]).to(model.device),  # [i0, i1, i0, i1]
            TEXT_INPUT_KEY: text_input,
        }  # [c0, c0, c1, c1]
        output = generate_output_blip(args, model, samples)
        curr_score = {
            "id": example["id"],
            "c0_i0": output[0],
            "c0_i1": output[1],
            "c1_i0": output[2],
            "c1_i1": output[3],
            "prompt": text_input,
        }
        predictions.append(curr_score)

    output_dir = get_dir_path(args)
    save_to_json(os.path.join(output_dir, "predictions.json"), predictions)  # for debugging
    eval_winoground(args, output_dir, predictions, template_expr)


def run_inference(args):
    """
    Performs Visual Question Answering (VQA) inference on the Winoground dataset using the BLIP2 model.
    The function supports three different formats for the VQA task:
        - basic_qa: the model is given an image and a question and it has to answer the question.
        - caption_qa: the model is given an image, a question, and a caption. The model has to answer the question,
          with the caption providing context for improving the answer.
        - decompose_qa: the model is given an image, a question, and a decomposition of the question. The model has to
          answer the question, with the decomposition providing context for improving the answer.

    Notes:
        - The function loads the BLIP2 model and any necessary pre-processors based on the arguments.
        - If the `vqa_format` argument is set to "caption_qa" and the `model_name` argument is set to "blip_vqa",
          the function loads a separate caption model and pre-processors.
        - If the `vqa_format` argument is set to "decompose_qa", the function calls `run_decompose_qa`,
          which doesn't return scores.
        - For all formats except "decompose_qa", the function calls `eval_winoground()` to
          compute and save the scores.
    """
    model, processor = load_model_and_processors(args.model_name, args.device)
    output_dir = get_dir_path(args)
    if os.path.exists(output_dir) and not args.overwrite_output_dir:
        logger.info(f"Output directory {output_dir} already exists. Skipping inference.")
        return

    logger.info(f'Selected prompt name :"{args.prompt_name}"')
    args.num_beams = 5
    args.max_length = 10

    run_vqa_inference_and_evaluation(args, model, processor)
