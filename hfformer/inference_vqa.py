# script supports both OKVQA, AOKVQA, Visual7W and VQAv2 datasets

import gc
import json
import os
import time
from typing import List
import torch.multiprocessing as mp

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          T5ForConditionalGeneration, T5Tokenizer)

from caption_generation import CaptionPrefixHandler, PromptcapPrefixHandler
from dataset_zoo.custom_dataset import VQADataset
from modeling_utils import (extract_rationale_per_question, get_dir_path,
                            get_prompt_handler, majority_vote_with_indices,
                            save_to_json, save_vqa_answers, save_vqa_answers_chunked, set_seed)
from rationale_generation import CoTPrefixHandler
from utils.globals import MODEL_CLS_INFO, VQA_PROMPT_COLLECTION
from utils.logger import Logger
from utils.okvqa_utils import (extract_answer_from_cot,
                               postprocess_batch_vqa_generation_blip2,
                               postprocess_ok_vqa_generation_flamingo)
from utils.vqa_accuracy import eval_aokvqa, eval_gqa, eval_visual7w, eval_vqa

set_seed(42)

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
    capgen.generate_caption_to_use_as_prompt_prefix(prompt_handler_2, args.split)

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
        cotgen.generate_cot_to_use_as_prompt_prefix(prompt_handler_2, args.split)
        cot_args["cot_gen_cls"] = cotgen

    return cot_args


def update_configs(args):
    args.num_beams = 5
    args.num_captions = 1
    args.max_length = 10
    args.length_penalty = -1.0
    args.no_repeat_ngram_size = 0

    if "xl" in args.model_name:
        args.batch_size = 128

    if args.dataset_name == "vqa_v2":
        args.batch_size = 256

    if "knn" in args.prompt_name:
        args.batch_size //= 2

    if "rationale" in args.prompt_name and ("mixer" not in args.prompt_name or "iterative" not in args.prompt_name):
        args.max_length = 100
        args.length_penalty = 1.4
        args.no_repeat_ngram_size = 3

    if args.self_consistency:
        args.num_beams = 1
        args.num_captions = 30
        args.temperature = 0.7
    
    args.batch_size = 32
    return args


def reset_configs(args):
    args.batch_size = 32
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
        additional_args = generate_and_cache_caption(args, prompt_handler_2)
    elif args.vqa_format == "cot_qa":
        if "mixer" in args.prompt_name:
            additional_args = use_ratione_as_caption(args, prompt_handler_2)
        else:
            additional_args = generate_and_cache_cot_rationale(args, prompt_handler_2)

    if "iterative" in args.prompt_name:
        args.max_length = 10
        args.length_penalty = -1.0
        args.no_repeat_ngram_size = 0

    model_name = MODEL_CLS_INFO["hfformer"][args.model_name]["name"]

    if args.model_name in ["flant5xl", "flant5xxl"]:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    elif args.model_name in ["opt27b", "opt67b"]:
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-2.7b")
        model = AutoModelForCausalLM.from_pretrained("facebook/opt-2.7b")
    else:
        processor = Blip2Processor.from_pretrained(model_name)
        if "opt" in args.model_name:
            processor.tokenizer.padding_side = "left"
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    model.eval()
    batch_size = args.batch_size
    num_workers = args.num_workers
    dataset_args = {
        "config": args,
        "dataset_name": args.dataset_name,
        "processor": processor,
        "prompt_handler": prompt_handler,
        "model_name": args.model_name,
        "split": args.split,
    }
    dataset_args.update(additional_args)
    dataset = VQADataset(**dataset_args)


    def collate_fn(batch):
        # get the keys from the first batch element
        batch_keys = batch[0].keys()
        bkeys = ["question", "answer", "question_id", "prompted_question", "image", "image_path"]

        # Create the batch dictionary
        processed_batch = {}
        for bkey in bkeys:
            if bkey in batch_keys:
                processed_batch[bkey + "s"] = [example[bkey] for example in batch]

        text_inputs = processor.tokenizer(
                        [example["prompted_question"] for example in batch], padding=True, return_tensors="pt"
            )
        
        processed_batch["pixel_values"] = torch.stack([example["pixel_values"] for example in batch])
        processed_batch["input_ids"] = text_inputs["input_ids"]
        processed_batch["attention_mask"] = text_inputs["attention_mask"]
        return processed_batch

    while True:
        try:
            # batch_size = get_optimal_batch_size(args)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                collate_fn=collate_fn,
                pin_memory=True,  # set to True to enable faster data transfer between CPU and GPU
            )

            logger.info(f" Num examples: \t{len(dataset)}")
            logger.info(f" Batch size: \t{batch_size}")
            logger.info(f" Num iterations: \t{len(dataset) // batch_size}")

            predictions = {}
            for batch in tqdm(dataloader, desc="Batch"):
                images = batch["images"]
                prompt = batch["prompted_questions"]

                if args.blind:  # zero out each images pixels, note that images are list of images
                    images = [TF.to_tensor(img) * 0 for img in images]

                # text_input = [txt_processors["eval"](q) for q in text_input]
                logger.debug(f"TEXT INPUT: {json.dumps(prompt, indent=2)}")

                if args.model_name in ["flant5xl", "flant5xxl"]:
                    inputs = tokenizer(prompt, padding=True, return_tensors="pt").to(model.device)
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_length,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                    output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                else:
                    # inputs = processor(images=images, text=prompt, padding=True, return_tensors="pt").to(
                    #     model.device, torch.bfloat16
                    # )
                    # print(inputs)
                    inputs = {
                        "input_ids": batch["input_ids"].to(model.device),
                        "attention_mask": batch["attention_mask"].to(model.device),
                        "pixel_values": batch["pixel_values"].to(model.device, torch.bfloat16),
                    }
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=args.max_length,
                        num_beams=args.num_beams,
                        length_penalty=args.length_penalty,
                        no_repeat_ngram_size=args.no_repeat_ngram_size,
                    )
                    output = processor.batch_decode(generated_ids, skip_special_tokens=True)

                if args.self_consistency:
                    logger.info(f"RAW PREDICTION: {json.dumps(output, indent=2)}")
                    if "rationale" in args.prompt_name:
                        batch["reasoning_paths"] = extract_rationale_per_question(output, args.num_captions)
                        extracted_answers = [extract_answer_from_cot(prediction) for prediction in output]
                        batch["reasoning_answers"] = extract_rationale_per_question(
                            extracted_answers, args.num_captions
                        )
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

                    # for question, pred in zip(batch["questions"], output):
                    #     logger.debug(f"QUESTION: {question} | PREDICTION: {pred}")

                for i, prediction in enumerate(output):
                    example = {}
                    example["raw_prediction"] = batch["raw_prediction"][i]
                    example["flamingo_processed_prediction"] = postprocess_ok_vqa_generation_flamingo(prediction)
                    # example["image"] = batch["images"][i]
                    example["question"] = batch["questions"][i]
                    example["prediction"] = batch["prediction"][i]
                    example["question_id"] = batch["question_ids"][i]
                    example["prompt"] = prompt[i]
                    example["answer"] = batch["answers"][i]
                    if "reasoning_paths" in batch:
                        example["reasoning_paths"] = batch["reasoning_paths"][i]
                    predictions[example["question_id"]] = example
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                batch_size = batch_size // 2
                logger.warning(f"CUDA OOM error. Reducing batch size to {batch_size}.")
                del dataloader
                continue
            else:
                raise e
        break

    output_dir = get_dir_path(args)
    output_dir = os.path.join(output_dir, kwargs.get("identifier", ""))  # for grid search
    if args.dataset_name == "vqa_v2":
        output_dir = os.path.join(output_dir, f"chunk{args.chunk_id}")  # for vqa_v2
    save_to_json(os.path.join(output_dir, "predictions.json"), predictions)  # for debugging

    if args.dataset_name in ["okvqa"]:
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


def run_inference(args):
    """
    Performs Visual Question Answering (VQA) inference on the OKVQA and VQAv2 dataset using the BLIP2 model.
    The function supports three different formats for the VQA task:
        - standard_qa: the model is given an image and a question and it has to answer the question.
        - caption_qa: the model is given an image, a question, and a caption. The model has to answer the question,
          with the caption providing context for improving the answer.
    """
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
        N = 20 # number of days
        if not args.overwrite_output_dir and os.path.exists(fpath):
            file_mod_time = os.path.getmtime(fpath)
            current_time = time.time()
            n_days_in_seconds = N * 24 * 60 * 60
            if current_time - file_mod_time < n_days_in_seconds:
                logger.info(f"File {fpath} already exists. Skipping inference.")
                continue

        args = update_configs(args)
        run_vqa_inference_and_evaluation(args)
        args = reset_configs(args)

        # Add the following two lines after each run
        torch.cuda.empty_cache()  # Release unused GPU memory
        gc.collect()  # Trigger Python garbage collection
