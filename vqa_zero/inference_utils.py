import json
import os
import time
from typing import List, Union

import numpy as np
import torch
from transformers import (AutoModelForCausalLM, AutoModelForVision2Seq,
                          AutoProcessor, AutoTokenizer,
                          Blip2ForConditionalGeneration, Blip2Processor,
                          T5ForConditionalGeneration, T5Tokenizer)

from utils.config import HUGGINFACE_HUB_DIR, OUTPUT_DIR, VQA_DATASET_DIR
from utils.globals import DATASET_CONFIG, MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.helpers import CustomJsonEncoder
from utils.logger import Logger

logger = Logger(__name__)


# model and inference related methods
def load_model_and_processors(model_cls_name: str, device: str, autocast_dtype):
    model_name = MODEL_CLS_INFO["hfformer"][model_cls_name]["name"]

    processor, tokenizer = None, None
    if model_cls_name in ["flant5xl", "flant5xxl"]:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=autocast_dtype, device_map="auto")

    elif model_cls_name in ["opt27b", "opt67b", "vicuna13b", "redpajama", "redpajama_instruct"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=autocast_dtype, device_map="auto")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

    elif model_cls_name == "kosmos2":
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        model = model.to(device)

    elif model_cls_name in ["open_flamingo_redpajama", "open_flamingo_redpajama_instruct", "open_flamingo_mpt"]:
        from open_flamingo import create_model_and_transforms

        lang_encoder_path = MODEL_CLS_INFO["hfformer"][model_cls_name]["lang_encoder_path"]
        tokenizer_path = MODEL_CLS_INFO["hfformer"][model_cls_name]["tokenizer_path"]
        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=lang_encoder_path,
            tokenizer_path=tokenizer_path,
            cross_attn_every_n_layers=1,
            cache_dir=HUGGINFACE_HUB_DIR,  # Defaults to ~/.cache
        )

        # grab model checkpoint from huggingface hub
        from huggingface_hub import hf_hub_download

        from dataset_zoo.custom_processor import FlamingoProcessor

        checkpoint_path = hf_hub_download("openflamingo/OpenFlamingo-3B-vitl-mpt1b", "checkpoint.pt")
        model.load_state_dict(torch.load(checkpoint_path), strict=False)
        model = model.to(device)
        tokenizer.padding_side = "left"
        processor = FlamingoProcessor(tokenizer, image_processor, device, autocast_dtype)

    elif model_cls_name == "llava":
        from dataset_zoo.custom_processor import LlaVaProcessor
        from LLaVA.llava.mm_utils import (KeywordsStoppingCriteria,
                                          get_model_name_from_path)
        from LLaVA.llava.model.builder import load_pretrained_model
        from LLaVA.llava.utils import disable_torch_init

        disable_torch_init()
        model_path = os.path.expanduser(model_name)
        model_name = get_model_name_from_path(model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = MODEL_CLS_INFO["hfformer"][model_cls_name]["base"]
        tokenizer_, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        tokenizer_.padding_side = "left"
        processor = LlaVaProcessor(tokenizer_, image_processor, model.config.mm_use_im_start_end)

    else:
        processor = Blip2Processor.from_pretrained(model_name)
        if "opt" in model_cls_name:
            processor.tokenizer.padding_side = "left"
        model = Blip2ForConditionalGeneration.from_pretrained(model_name, torch_dtype=autocast_dtype, device_map="auto")

    return model, processor, tokenizer


def get_optimal_batch_size_v2(model, seq_length: int, batch_size: int):
    """
    Returns the optimal batch size for inference that considers the model size and the available GPU memory.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_memory = torch.cuda.get_device_properties(device).total_memory / 1024 / 1024 / 1024  # in GB
    num_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_size = num_parameters * 4 / 1024 / 1024 / 1024  # in GB

    print_str = f"Initial batch size: {batch_size}, "
    print_str += f"GPU memory: {gpu_memory:.2f} GB, "
    print_str += f"Model size: {model_size:.2f} GB, "
    print_str += f"Sequence length: {seq_length}, "

    # Compute the memory usage per sample
    sample_memory = model_size + seq_length * 4 / 1024 / 1024 / 1024  # in GB
    print_str += f"Memory per sample: {sample_memory:.2f} GB"

    # Compute the optimal batch size
    max_batch_size = int(gpu_memory / sample_memory)
    optimal_batch_size = min(max_batch_size, batch_size)

    print_str += f", Max batch size: {max_batch_size}, "
    print_str += f"Optimal batch size: {optimal_batch_size}"

    logger.info(print_str)
    return optimal_batch_size


def generate_output_blip(args, model, samples):
    with torch.no_grad(), torch.cuda.amp.autocast():
        if args.model_name == "blip_vqa":
            return model.predict_answers(samples=samples, inference_method="generate")
        else:
            return model.generate(
                samples=samples,
                num_beams=args.num_beams,
                max_length=args.max_length,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                num_captions=args.num_captions,
                use_nucleus_sampling=True if args.self_consistency else False,
                temperature=0.7 if args.self_consistency else 1.0,
            )


# prompt related methods
def get_prompt_template_handler(args, prompt_name: str = None):
    if prompt_name is None and args.prompt_name is not None:
        prompt_name = args.prompt_name

    if prompt_name is None:
        raise ValueError(f"prompt_name should be provided. Got {prompt_name}")

    if len(prompt_name.split(",")) > 1:
        prompt_name = prompt_name.split(",")

    if args.vqa_format == "standard_vqa":
        handler = PromptingHandler(args.dataset_name, prompt_name, subset_name="vqa")
        template_expr = handler.prompt.jinja if handler.prompt else ""
        logger.info(f"PROMPT TEMPLATE: {template_expr}")

    elif args.vqa_format == "caption_vqa":
        if len(prompt_name) != 2:
            raise ValueError(f"Prompt_name should be a list of two prompts for caption_qa format. Got {prompt_name}")
        vqa_template_handler = PromptingHandler(args.dataset_name, prompt_name[0], subset_name="vqa")
        template_expr_vqa = vqa_template_handler.prompt.jinja if vqa_template_handler.prompt else ""
        logger.info(f"PROMPT TEMPLATE: {template_expr_vqa}")
        context_template_handler = PromptingHandler(args.dataset_name, prompt_name[1], subset_name="captioning")
        template_expr_context = context_template_handler.prompt.jinja if context_template_handler.prompt else ""
        logger.info(f"PROMPT TEMPLATE: {template_expr_context}")
        handler = [vqa_template_handler, context_template_handler]

    elif args.vqa_format == "cot_vqa":
        if len(prompt_name) != 2:
            raise ValueError(f"Prompt_name should be a list of two prompts for caption_qa format. Got {prompt_name}")
        vqa_template_handler = PromptingHandler(args.dataset_name, prompt_name[0], subset_name="vqa")
        template_expr_vqa = vqa_template_handler.prompt.jinja if vqa_template_handler.prompt else ""
        logger.info(f"PROMPT TEMPLATE: {template_expr_vqa}")
        context_template_handler = PromptingHandler(args.dataset_name, prompt_name[1], subset_name="vqa")
        template_expr_context = context_template_handler.prompt.jinja if context_template_handler.prompt else ""
        logger.info(f"PROMPT TEMPLATE: {template_expr_context}")
        handler = [vqa_template_handler, context_template_handler]

    else:
        raise NotImplementedError(f"VQA format {args.vqa_format} not implemented")

    template_expr = (
        template_expr_context + " [SEP] " + template_expr_vqa
        if args.vqa_format in ["caption_vqa", "cot_vqa"]
        else template_expr
    )
    return handler, template_expr


def apply_prompt_to_example_winoground(handler: PromptingHandler, example):
    return (
        handler.gpt3_baseline_qa_prompting_handler_for_winoground(example)
        if handler.prompt_name.startswith("prefix_")  # means it's a cached question prompt
        else handler.generic_prompting_handler_for_winoground(example)
    )


def is_vqa_output_cache_exists(args):
    output_dir = get_output_dir_path(args)

    fn_suffix = "predictions.json"
    fpath = os.path.join(output_dir, fn_suffix)
    if not args.overwrite_output_dir and os.path.exists(fpath):
        logger.info(f"File {fpath} already exists. Skipping inference.")
        return True

    return False


def get_output_dir_path(args, **kwargs):
    required_args = [args.dataset_name, args.model_name, args.vqa_format, args.prompt_name]
    if any(val is None for val in required_args):
        raise ValueError(
            f"Please provide `dataset_name`, `model_name`, `vqa_format` and `prompt_name` to get the output directory path. "
            f"Provided: {required_args}"
        )

    # Construct model_name_str based on arguments
    model_name_str = args.model_name
    if args.gen_model_name and args.model_name != args.gen_model_name:
        model_name_str = f"{model_name_str}+{args.gen_model_name}"

    if args.blind:
        model_name_str = f"{model_name_str}+blind"

    few_shot_str = "few_shot" if args.few_shot else ""

    prompt_name_str = "/".join(args.prompt_name.split(","))
    path_components = [
        OUTPUT_DIR,
        "output",
        args.dataset_name,
        model_name_str,
        few_shot_str,
        args.vqa_format,
        prompt_name_str,
        args.task_type,
        args.split,
    ]

    if args.self_consistency:
        path_components.append("self_consistency")

    if kwargs.get("identifier"):
        path_components.append(kwargs.get("identifier"))

    if args.chunk_id is not None:
        path_components.append("chunked")
        path_components.append(f"chunk{args.chunk_id}")

    # Join all path components together
    dir_path = os.path.join(*path_components)
    return dir_path


def save_to_json(json_path: str, all_results):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, cls=CustomJsonEncoder, indent=2)

    logger.info(f"Saved the output to {json_path}")


def save_vqa_answers(output_dir: str, dataset_name: str, predictions, vicuna_ans_parser: bool = False):
    """
    Saves the predictions in a format that is compatible with the VQA accuracy computation script.
    """
    annotation_file = DATASET_CONFIG[dataset_name]["val"]["annotation_file"]
    annotation_data_path = os.path.join(VQA_DATASET_DIR, "datasets", dataset_name, annotation_file)
    with open(annotation_data_path) as f:
        answer_data = json.load(f)

    # Infer the type of question_id from the type of the first key in predictions
    converter = str if isinstance(next(iter(predictions)), str) else int

    # Use a new list to store annotations that have a matching question_id in predictions
    new_annotations = []

    for ann in answer_data["annotations"]:
        question_id = converter(ann["question_id"])

        if question_id not in predictions:
            logger.debug("warning: missing question_id: %s" % question_id)
            continue

        pred_key = "raw_prediction" if dataset_name == "vqa_v2" and not vicuna_ans_parser else "prediction"
        answer = predictions[question_id][pred_key]
        ann["answer"] = answer

        # Add the annotation to the new list
        new_annotations.append(ann)

    answer_fpath = os.path.join(output_dir, "annotations+vqa_answers.json")
    save_to_json(answer_fpath, new_annotations)


def save_predictions(output_dir: str, predictions):
    with open(os.path.join(output_dir, "predictions.json"), "w") as f:
        json.dump(predictions, f, indent=2)


def save_vqa_answers_aggregate_chunks(output_dir: str, dataset_name: str, vicuna_ans_parser: bool = False):
    """
    Saves the predictions in a format that is compatible with the VQA accuracy computation script.
    """

    output_dir = os.path.dirname(output_dir)  # remove the chunked_[id] directory
    chunked_filepaths = [
        os.path.join(output_dir, dir_name, "predictions.json")
        for dir_name in os.listdir(output_dir)
        if os.path.exists(os.path.join(output_dir, dir_name, "predictions.json"))
    ]

    logger.info(f"Found {len(chunked_filepaths)} chunked files")

    predictions = {}
    for filepath in chunked_filepaths:
        with open(filepath, "r") as f:
            curr_predictions = json.load(f)
        predictions.update(curr_predictions)

    annotation_file = DATASET_CONFIG[dataset_name]["val"]["annotation_file"]
    annotation_data_path = os.path.join(VQA_DATASET_DIR, "datasets", dataset_name, annotation_file)
    with open(annotation_data_path) as f:
        answer_data = json.load(f)

    # Infer the type of question_id from the type of the first key in predictions
    converter = str if isinstance(next(iter(predictions)), str) else int

    # Use a new list to store annotations that have a matching question_id in predictions
    new_annotations = []

    for ann in answer_data["annotations"]:
        question_id = converter(ann["question_id"])

        if question_id not in predictions:
            logger.debug("warning: missing question_id: %s" % question_id)
            continue

        pred_key = "raw_prediction" if dataset_name == "vqa_v2" and not vicuna_ans_parser else "prediction"
        answer = predictions[question_id][pred_key]
        ann["answer"] = answer

        # Add the annotation to the new list
        new_annotations.append(ann)

    # save only the total predictions match the original annotations
    if len(new_annotations) == len(answer_data["annotations"]):
        prediction_fpath = os.path.join(output_dir, "predictions.json")
        answer_fpath = os.path.join(output_dir, "annotations+vqa_answers.json")
        save_to_json(prediction_fpath, predictions)
        save_to_json(answer_fpath, new_annotations)


def save_gqa_answers(output_dir: str, dataset_name: str, predictions):
    """
    Saves the predictions in a format that is compatible with the VQA accuracy computation script.
    """
    annotation_file = DATASET_CONFIG[dataset_name]["annotation_file"]
    annotation_data_path = os.path.join(VQA_DATASET_DIR, "datasets", dataset_name, annotation_file)
    question_answer_data = {}
    with open(annotation_data_path) as f:
        answer_data = json.load(f)

    for question_id, ann in answer_data.items():
        answer = predictions[question_id]["prediction"]
        question_answer_data.update(
            {
                "question_id": question_id,
                "predicted_answer": answer,
                "question": ann["question"],
                "answer": ann["answer"],
            }
        )

    answer_fpath = os.path.join(output_dir, "annotations+vqa_answers.json")
    save_to_json(answer_fpath, question_answer_data)


def set_seed(seed_value):
    """Set seed value for reproducibility in PyTorch and NumPy"""
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)


def get_autocast_dtype():
    """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def format_last_predictions(
    questions: List[str],
    generated_outputs: Union[List[str], List[List[str]]],
    num_return_sequences: int,
    num_last_items=5,
):
    """
    Extract and format the last few predictions.

    Parameters:
    - questions (list): List of question strings.
    - generated_outputs (list): List of generated output strings.
    - num_last_items (int): Number of last items to retrieve. Default is 5.

    Returns:
    - formatted_messages (list): List of formatted prediction strings.
    """
    last_questions = questions[-num_last_items:]
    last_generated_outputs = [
        generated_outputs[i : i + num_return_sequences]
        for i in range(-num_last_items * num_return_sequences, 0, num_return_sequences)
    ]

    formatted_messages = []
    for question, outputs in zip(last_questions, last_generated_outputs):
        for pred in outputs:
            formatted_messages.append(f"question = {question}, prediction = {pred}")

    return "\n".join(formatted_messages)
