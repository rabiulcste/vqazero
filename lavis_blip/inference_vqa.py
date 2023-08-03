# script supports both OKVQA, AOKVQA, Visual7W and VQAv2 datasets

import json
import os
import time
import warnings
from typing import List

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from caption_generation import CaptionPrefixHandler, PromptcapPrefixHandler
from dataset_zoo.custom_dataset import VQADataset, collate_fn
from modeling_utils import (extract_rationale_per_question,
                            generate_output_blip, get_dir_path,
                            get_optimal_batch_size, get_optimal_batch_size_v2,
                            get_prompt_handler, load_model_and_processors,
                            majority_vote_with_indices, save_gqa_answers,
                            save_to_json, save_vqa_answers, set_seed)
from rationale_generation import CoTPrefixHandler
from utils.globals import VQA_GRID_SEARCH, VQA_PROMPT_COLLECTION
from utils.logger import Logger
from utils.okvqa_utils import (extract_answer_from_cot,
                               postprocess_batch_vqa_generation_blip2,
                               postprocess_ok_vqa_generation_flamingo)
from utils.vqa_accuracy import eval_aokvqa, eval_gqa, eval_visual7w, eval_vqa

warnings.filterwarnings("default", category=UserWarning, module="transformers")

set_seed(42)

DATASET_CLASSES = {
    "visual7w": VQADataset,
    "okvqa": VQADataset,
    "vqa_v2": VQADataset,
    "aokvqa": VQADataset,
    "gqa": VQADataset,
    "carets": VQADataset,
}

data_splits_carets = [
    "antonym_consistency",
    "ontological_consistency",
    "phrasal_invariance",
    "symmetry_invariance",
    "negation_consistency",
]

logger = Logger(__name__)


def generate_and_cache_caption(args, prompt_handler_2):
    if args.gen_model_name is None:
        args.gen_model_name = args.model_name

    if args.gen_model_name == "promptcap":
        capgen = PromptcapPrefixHandler(args, device="cuda")
    else:
        capgen = CaptionPrefixHandler(args, device="cuda")

    # if the prompt requires similar samples in few-shot, then wil generate caption for training set too
    if "knn" in args.prompt_name:
        capgen.generate_caption_to_use_as_prompt_prefix(prompt_handler_2, "train")
    capgen.generate_caption_to_use_as_prompt_prefix(prompt_handler_2)

    caption_args = {
        "caption_gen_cls": capgen,
    }

    return caption_args


def use_ratione_as_caption(args, prompt_handler_2):
    if args.gen_model_name is None:
        args.gen_model_name = args.model_name
    cotgen = CoTPrefixHandler(args, device="cuda")

    cotgen.generate_cot_to_use_as_prompt_prefix(prompt_handler_2)
    cot_args = {"cot_gen_cls": cotgen}
    return cot_args


def generate_and_cache_cot_rationale(args, prompt_handler_2):
    if args.gen_model_name is None:
        args.gen_model_name = args.model_name
    cotgen = CoTPrefixHandler(args, device="cuda")

    cot_args = {}
    if "knn" in args.prompt_name:
        cotgen.generate_cot_to_use_as_prompt_prefix(prompt_handler_2, "train")

    if "iterative" in args.prompt_name:
        cotgen.generate_cot_to_use_as_prompt_prefix(prompt_handler_2)
        cot_args["cot_gen_cls"] = cotgen

    return cot_args


def update_configs(args):
    args.num_beams = 5
    args.num_captions = 1

    if "xxl" in args.model_name:
        args.batch_size = 48

    if "knn" in args.prompt_name:
        args.batch_size = 32
        if "xxl" in args.model_name:
            args.batch_size = 24

    if "rationale" in args.prompt_name and ("mixer" not in args.prompt_name or "iterative" not in args.prompt_name):
        args.max_length = 100
        args.length_penalty = 1.4
        args.no_repeat_ngram_size = 3
    else:
        args.max_length = 10
        args.length_penalty = -1.0
        args.no_repeat_ngram_size = 0

    if args.self_consistency:
        args.batch_size = 32
        args.num_beams = 1
        args.num_captions = 30
        args.temperature = 0.7
    return args


def run_vqa_inference_and_evaluation(args, **kwargs):
    if args.vqa_format not in ["basic_qa", "caption_qa", "cot_qa"]:
        raise NotImplementedError(
            f"Provided VQA format {args.vqa_format} is either not implemented yet or invalid argument provided."
        )

    # get prompting handler
    prompt_handler, template_expr = get_prompt_handler(args)

    if isinstance(prompt_handler, List):
        prompt_handler, prompt_handler_2 = prompt_handler

    additional_args = {}
    if args.vqa_format == "caption_qa":
        # here, we will generate the caption to use as prompt prefix if not already generated
        # the caption_gen_cls is a CaptionPrefixHandler object which is provided to the dataset class \
        # for loading the cached caption from disk to use as prompt prefix
        additional_args = generate_and_cache_caption(args, prompt_handler_2)
    elif args.vqa_format == "cot_qa":
        if "mixer" in args.prompt_name:
            additional_args = use_ratione_as_caption(args, prompt_handler_2)
        else:
            additional_args = generate_and_cache_cot_rationale(args, prompt_handler_2)
    # TODO: this should be moved to cot_qa as well

    if "iterative" in args.prompt_name:
        args.max_length = 20
        args.length_penalty = -1.0
        args.no_repeat_ngram_size = 0

    model, processor = load_model_and_processors(args.model_name, args.device)
    if isinstance(processor, tuple):
        vis_processors, txt_processors = processor

    # batch_size = get_optimal_batch_size(args)
    batch_size = get_optimal_batch_size_v2(model=model, seq_length=512, batch_size=args.batch_size)
    dataset_class = DATASET_CLASSES[args.dataset_name]
    dataset_args = {
        "config": args,
        "dataset_name": args.dataset_name,
        "vis_processors": vis_processors,
        "prompt_handler": prompt_handler,
        "model_name": args.model_name,
        "split": args.split,
    }
    dataset_args.update(additional_args)
    dataset = dataset_class(**dataset_args)
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

    TEXT_INPUT_KEY = "text_input" if args.model_name == "blip_vqa" else "prompt"
    predictions = {}
    for batch in tqdm(dataloader, desc="Batch"):
        # questions = [txt_processors["eval"](q) for q in questions]
        images = batch["images"]
        image_tensors = [vis_processors["eval"](image) for image in images]
        text_input = batch["prompted_questions"]
        # text_input = [txt_processors["eval"](q) for q in text_input]
        logger.debug(f"TEXT INPUT: {json.dumps(text_input, indent=2)}")

        samples = {
            "image": torch.stack(image_tensors).to(model.device),
            TEXT_INPUT_KEY: text_input,
        }
        output = generate_output_blip(args, model, samples)

        if args.self_consistency:
            logger.info(f"RAW PREDICTION: {json.dumps(output, indent=2)}")
            if "rationale" in args.prompt_name:
                batch["reasoning_paths"] = extract_rationale_per_question(output, args.num_captions)
                extracted_answers = [extract_answer_from_cot(prediction) for prediction in output]
                batch["reasoning_answers"] = extract_rationale_per_question(extracted_answers, args.num_captions)
            else:
                extracted_answers = list(output)
            processed_answers = postprocess_batch_vqa_generation_blip2(args.dataset_name, extracted_answers)
            majority_answers, indices = majority_vote_with_indices(processed_answers, args.num_captions)
            batch["raw_prediction"] = [output[i] for i in indices]
            batch["prediction"] = majority_answers
            output = batch["raw_prediction"]

            for question, pred in zip(batch["questions"], output):
                logger.info(f"QUESTION: {question} | PREDICTION: {pred}")
        else:
            batch["raw_prediction"] = output
            if "rationale" in args.prompt_name and "iterative" not in args.prompt_name:
                output = [extract_answer_from_cot(prediction) for prediction in output]

            if args.dataset_name in ["aokvqa", "visual7w"] and args.task_type == "multiple_choice":
                batch["prediction"] = output
            else:
                batch["prediction"] = postprocess_batch_vqa_generation_blip2(
                    args.dataset_name, output, batch["questions"]
                )  # answer batch processing for blip2

            for question, pred in zip(batch["questions"], output):
                logger.debug(f"QUESTION: {question} | PREDICTION: {pred}")

        for i, prediction in enumerate(output):
            example = {}
            example["raw_prediction"] = batch["raw_prediction"][i]
            example["flamingo_processed_prediction"] = postprocess_ok_vqa_generation_flamingo(prediction)
            # example["image"] = batch["images"][i]
            example["question"] = batch["questions"][i]
            example["prediction"] = batch["prediction"][i]
            example["question_id"] = batch["question_ids"][i]
            example["prompt"] = text_input[i]
            example["answer"] = batch["answers"][i]
            if "reasoning_paths" in batch:
                example["reasoning_paths"] = batch["reasoning_paths"][i]
            predictions[example["question_id"]] = example

    output_dir = get_dir_path(args)
    output_dir = os.path.join(output_dir, kwargs.get("identifier", ""))  # for grid search
    save_to_json(os.path.join(output_dir, "predictions.json"), predictions)  # for debugging

    if args.dataset_name in ["vqa_v2", "okvqa"]:
        save_vqa_answers(output_dir, args.dataset_name, predictions)  # for VQA accuracy computation
        eval_vqa(args, output_dir)
    elif args.dataset_name == "gqa":
        # save_gqa_answers(output_dir, args.dataset_name, predictions)  # for VQA accuracy computation
        eval_gqa(args, output_dir)
    elif args.dataset_name == "visual7w":
        eval_visual7w(args, output_dir, predictions, multiple_choice=True)  # TODO: make this nicer and more flexible
    elif args.dataset_name == "aokvqa":
        multiple_choice = True if args.task_type == "multiple_choice" else False
        eval_aokvqa(args, output_dir, predictions, multiple_choice)  # TODO: make this nicer and more flexible


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
    """
    Performs Visual Question Answering (VQA) inference on the OKVQA and VQAv2 dataset using the BLIP2 model.
    The function supports three different formats for the VQA task:
        - basic_qa: the model is given an image and a question and it has to answer the question.
        - caption_qa: the model is given an image, a question, and a caption. The model has to answer the question,
          with the caption providing context for improving the answer.
    """
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

        output_dir = get_dir_path(args)

        fpath = os.path.join(output_dir, "result_meta.json")
        N = 4
        if not args.overwrite_output_dir and os.path.exists(fpath):
            file_mod_time = os.path.getmtime(fpath)
            current_time = time.time()
            n_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < n_days_in_seconds:
                logger.info(f"File {fpath} already exists. Skipping inference.")
                continue

        args = update_configs(args)

        if args.dataset_name == "carets":  # carets is a special case with multiple data split
            for split in data_splits_carets:
                logger.info(f"Running inference on {split} split.")
                run_vqa_inference_and_evaluation(args, split=split)
        else:
            run_vqa_inference_and_evaluation(args)
