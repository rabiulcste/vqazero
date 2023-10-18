import json
import os
import warnings
from typing import List

import torch
from dataset_zoo.custom_dataset import VQADataset, collate_fn
from evals.answer_postprocess import answer_postprocess_batch
from evals.vqa_accuracy import eval_aokvqa, eval_gqa, eval_visual7w, eval_vqa
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.globals import VQA_GRID_SEARCH, VQA_PROMPT_COLLECTION
from utils.helpers import is_vqa_output_cache_exists, update_configs
from utils.logger import Logger
from utils.okvqa_utils import postprocess_ok_vqa_generation_flamingo
from vqa_zero.caption_generation import CaptionGenerator, PromptCapGenerarator
from vqa_zero.inference_utils import (generate_output_blip,
                                     get_optimal_batch_size,
                                     get_optimal_batch_size_v2,
                                     get_output_dir_path,
                                     get_prompt_template_handler,
                                     load_model_and_processors,
                                     save_gqa_answers, save_to_json,
                                     save_vqa_answers, set_seed)
from vqa_zero.rationale_generation import ChainOfThoughtGenerator

warnings.filterwarnings("default", category=UserWarning, module="transformers")

set_seed(42)

logger = Logger(__name__)


def generate_context_cache_if_not_exists(args, context_handler):
    """
    Generate context based on the given vqa_format and gen_model_name from the args.

    If the vqa_format is 'caption_vqa':
        - It either generates captions using PromptCapGenerarator if gen_model_name is 'promptcap',
        or uses CaptionGenerator for other gen_model_names.

    If the vqa_format is 'cot_qa':
        - It generates a chain-of-thought rationale using ChainOfThoughtGenerator.

    For both formats:
        - Context is generated for both 'train' and 'val' splits if 'knn' is in the prompt_name,
        otherwise, only for the 'val' split.

    Raises:
        ValueError: If an unsupported vqa_format is provided.
    """
    args.gen_model_name = args.model_name if args.gen_model_name is None else args.gen_model_name
    split_names = ["train", "val"] if "knn" in args.prompt_name else ["val"]

    # Decide the generator and action based on the vqa_format and gen_model_name
    if args.vqa_format == "caption_vqa":
        if args.gen_model_name == "promptcap":
            generator = PromptCapGenerarator(args, device="cuda")
            action = generator.generate_caption
        else:
            generator = CaptionGenerator(args, device="cuda")
            action = generator.generate_caption
    elif args.vqa_format == "cot_vqa":
        generator = ChainOfThoughtGenerator(args, device="cuda")
        action = generator.generate_chain_of_thought_rationale
    else:
        raise ValueError(f"Unsupported vqa_format: {args.vqa_format}")

    # Perform the action for each split
    for split_name in split_names:
        action(context_handler, split_name)


def run_vqa_inference_and_evaluation(args, **kwargs):
    vqa_prompt_handler, template_expr = get_prompt_template_handler(args)

    if isinstance(vqa_prompt_handler, List):
        vqa_prompt_handler, context_handler = vqa_prompt_handler
        generate_context_cache_if_not_exists(args, context_handler)

    model, processor = load_model_and_processors(args.model_name, args.device)
    if isinstance(processor, tuple):
        vis_processors, txt_processors = processor

    batch_size = get_optimal_batch_size_v2(model=model, seq_length=512, batch_size=args.batch_size)
    dataset_args = {
        "config": args,
        "dataset_name": args.dataset_name,
        "vis_processors": vis_processors,
        "prompt_handler": vqa_prompt_handler,
        "model_name": args.model_name,
        "split": args.split,
    }
    dataset = VQADataset(**dataset_args)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,  # set to True to enable faster data transfer between CPU and GPU
    )

    logger.info(f" Num examples: \t{len(dataset)}")
    logger.info(f" Batch size: \t{batch_size}")
    logger.info(f" Num iterations: \t{len(dataset) // args.batch_size}")

    text_input_key = "text_input" if args.model_name == "blip_vqa" else "prompt"
    predictions = {}
    for batch in tqdm(dataloader, desc="Batch"):
        images = batch["images"]
        image_tensors = [vis_processors["eval"](image) for image in images]
        text_input = batch["prompted_questions"]
        logger.debug(f"TEXT INPUT: {json.dumps(text_input, indent=2)}")

        samples = {
            "image": torch.stack(image_tensors).to(model.device),
            text_input_key: text_input,
        }
        output = generate_output_blip(args, model, samples)
        batch, output = answer_postprocess_batch(args, batch, output, logger)

        for i, prediction in enumerate(output):
            example = {}
            example["raw_prediction"] = batch["raw_prediction"][i]
            example["flamingo_processed_prediction"] = postprocess_ok_vqa_generation_flamingo(prediction)
            example["question"] = batch["questions"][i]
            example["prediction"] = batch["prediction"][i]
            example["question_id"] = batch["question_ids"][i]
            example["prompt"] = text_input[i]
            example["answer"] = batch["answers"][i]
            if "reasoning_paths" in batch:
                example["reasoning_paths"] = batch["reasoning_paths"][i]

            predictions[example["question_id"]] = example

    output_dir = get_output_dir_path(args)
    output_dir = os.path.join(output_dir, kwargs.get("identifier", ""))  # for grid search
    save_to_json(os.path.join(output_dir, "predictions.json"), predictions)  # for debugging

    multiple_choice = True if args.task_type == "multiple_choice" else False
    if args.dataset_name in ["vqa_v2", "okvqa"]:
        save_vqa_answers(output_dir, args.dataset_name, predictions)  # for VQA accuracy computation
        eval_vqa(args, output_dir)
    elif args.dataset_name == "gqa":
        save_gqa_answers(output_dir, args.dataset_name, predictions)  # for VQA accuracy computation
        eval_gqa(args, output_dir)
    elif args.dataset_name == "visual7w":
        eval_visual7w(args, output_dir, predictions, multiple_choice)
    elif args.dataset_name == "aokvqa":
        eval_aokvqa(args, output_dir, predictions, multiple_choice)


def find_best_decoding_strategy(args):
    pair_grid = [
        (num_beams, max_length)
        for num_beams in VQA_GRID_SEARCH["num_beams"]
        for max_length in VQA_GRID_SEARCH["max_length"]
    ]
    for num_beams, max_length in pair_grid:
        args.num_beams = num_beams
        args.max_length = max_length
        identifier = f"num_beams={num_beams}_max_length={max_length}" if args.grid_search else ""
        run_vqa_inference_and_evaluation(args, identifier)


def run_inference(args):
    # in case, you want to do grid search to find best decoding stragety on a given prompt
    if args.grid_search and not args.prompt_name:
        raise ValueError("Grid search is only supported for a single prompt.")
    elif args.grid_search:
        find_best_decoding_strategy(args)

    all_prompts = [args.prompt_name]
    # in case, you want to evaluate all prompts in the collection
    if not args.prompt_name:  # if prompt name is not provided, we look for a list of prompts in the globals.py file
        caption_prompts = VQA_PROMPT_COLLECTION[args.dataset_name]["caption"]
        question_prompts = VQA_PROMPT_COLLECTION[args.dataset_name]["question"]
        cq_prompts = [f"{q},{c}" for q in question_prompts for c in caption_prompts]
        all_prompts = cq_prompts if args.vqa_format == "caption_qa" else question_prompts

    logger.info(f"Total prompts: {len(all_prompts)} will be evaluated.")

    for prompt_name in all_prompts:
        args.prompt_name = prompt_name
        logger.info(f'Selected prompt name :"{args.prompt_name}"')

        if is_vqa_output_cache_exists():
            continue

        args = update_configs(args)
        run_vqa_inference_and_evaluation(args)
