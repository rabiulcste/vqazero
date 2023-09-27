import json
import os

import numpy as np
import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          CLIPModel, CLIPProcessor)

from utils.config import OUTPUT_DIR, VQA_DATASET_DIR
from utils.globals import DATASET_CONFIG, MODEL_CLS_INFO
from utils.handler import PromptingHandler
from utils.logger import Logger

logger = Logger(__name__)


# model and inference related methods
def load_model_and_processors(model_cls_name: str, device):
    if model_cls_name.startswith("blip"):
        MODEL_INFO = MODEL_CLS_INFO["lavis"]
    else:
        MODEL_INFO = MODEL_CLS_INFO["hfformer"]

    if model_cls_name not in MODEL_INFO:
        raise ValueError(f"Invalid `args.model_name`. Provided: {model_cls_name}")

    logger.info(f"loading model and processor for `{model_cls_name}`")
    model_info = MODEL_INFO[model_cls_name]
    model_name = model_info["name"]
    model_type = model_info.get("model_type")

    if model_cls_name.startswith("blip"):
        from lavis.models import \
            load_model_and_preprocess  # this is due to the case that lavis conflict with recent transformers version

        model, vis_processors, txt_processors = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        processor = (
            vis_processors,
            txt_processors,
        )  # processor is a tuple of image and text processor for lavis models
    elif model_cls_name == "clip":
        model = CLIPModel.from_pretrained(model_name)
        processor = CLIPProcessor.from_pretrained(model_name)
    elif model_cls_name.startswith("git"):
        model = AutoModelForCausalLM.from_pretrained(model_name)
        processor = AutoProcessor.from_pretrained(model_name, padding_side="left")
    elif model_cls_name == "ofa_vqa":
        model = AutoModel.from_pretrained(model_name, use_cache=False)
        processor = AutoTokenizer.from_pretrained(model_name)
    else:
        from transformers import AutoModelForSeq2SeqLM, AutoProcessor

        processor = AutoProcessor.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    model = model.to(device)
    return model, processor


def get_optimal_batch_size_v2(model, seq_length, batch_size):
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


def get_image_tensors_winoground(example, vis_processors):
    NUM_IMAGES = 2
    return [vis_processors["eval"](example[f"image_{i}"].convert("RGB")) for i in range(NUM_IMAGES)]


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
        
    if args.dataset_name == "vqa_v2" and args.chunk_id is not None:
        path_components.append(f"chunk{args.chunk_id}")

    # Join all path components together
    dir_path = os.path.join(*path_components)
    return dir_path


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, "tolist"):  # Convert tensors to lists
            return obj.tolist()
        elif hasattr(obj, "name"):  # Convert PyTorch device to its name string
            return obj.name
        elif isinstance(obj, type):  # If it's a type/class object
            return str(obj)
        elif isinstance(obj, torch.device):  # Handling torch.device objects
            return str(obj)
        # You can add more custom handlers here if needed
        return super().default(obj)


def save_to_json(json_path, all_results):
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(all_results, f, cls=CustomEncoder, indent=2)

    logger.info(f"Saved the output to {json_path}")


def save_vqa_answers(output_dir, dataset_name, predictions, vicuna_ans_parser=False):
    """
    Saves the predictions in a format that is compatible with the VQA accuracy computation script.
    """
    annotation_file = DATASET_CONFIG[dataset_name]["val"]["annotation_file"]
    annotation_data_path = os.path.join(VQA_DATASET_DIR, "datasets", dataset_name, annotation_file)
    with open(annotation_data_path) as f:
        answer_data = json.load(f)

    # Infer the type of question_id from the type of the first key in predictions
    converter = str if isinstance(next(iter(predictions)), str) else int

    for ann in answer_data["annotations"]:
        question_id = ann["question_id"]
        question_id = converter(ann["question_id"])

        if question_id not in predictions:
            logger.info("warning: missing question_id: %s" % question_id)
            continue
        
        pred_key = "raw_prediction" if dataset_name == "vqa_v2" and not vicuna_ans_parser else "prediction"
        answer = predictions[question_id][pred_key]
        ann["answer"] = answer

    answer_fpath = os.path.join(output_dir, "annotations+vqa_answers.json")
    save_to_json(answer_fpath, answer_data["annotations"])


def save_vqa_answers_chunked(output_dir, chunk_id, predictions):
    """
    Saves the predictions in a format that is compatible with the VQA accuracy computation script.
    """
    answer_fpath = os.path.join(output_dir, f"prediction_chunk{chunk_id}.json")
    save_to_json(answer_fpath, predictions)


def save_gqa_answers(output_dir, dataset_name, predictions):
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
