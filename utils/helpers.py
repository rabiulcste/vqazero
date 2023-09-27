import gc
import os
import time
from collections import Counter

import torch

from utils.globals import VQA_PROMPT_COLLECTION, promptcap_prompts
from utils.logger import Logger
from vqa_zero.inference_utils import get_output_dir_path

logger = Logger(__name__)


def singularize_key(key):
    """Remove the trailing 's' from the key."""
    return key[:-1] if key.endswith("s") else key


def update_configs(args):
    # Base configuration
    config_updates = {
        "num_beams": 3,
        "num_captions": 1,
        "max_length": 10,
        "length_penalty": -1.0,
        "no_repeat_ngram_size": 0,
        "batch_size": 64,
    }

    # Update configs based on model_name
    if "xxl" in args.model_name or "kosmos" in args.model_name:
        config_updates["batch_size"] = 32

    # Update configs based on answer parser
    if args.vicuna_ans_parser:
        config_updates.update(
            {
                "max_length": 30,
            }
        )

    if "knn" in args.prompt_name:
        config_updates.update(
            {
                "max_length": 10,
            }
        )

    # Update configs based on prompt_name
    if "rationale" in args.prompt_name and ("mixer" not in args.prompt_name or "iterative" not in args.prompt_name):
        config_updates.update({"max_length": 100, "length_penalty": 1.0, "no_repeat_ngram_size": 3, "batch_size": 32})

    if "iterative" in args.prompt_name:
        config_updates.update(
            {
                "max_length": 10,
                "length_penalty": -1.0,
            }
        )

    # Update configs based on self_consistency
    if args.self_consistency:
        config_updates.update({"num_beams": 1, "num_captions": 30, "temperature": 0.7})

    # Apply updates to args
    for key, value in config_updates.items():
        setattr(args, key, value)

    return args


def is_vqa_output_cache_exists(args):
    N = 5
    output_dir = get_output_dir_path(args)

    fpath = os.path.join(output_dir, "result_meta.json")
    if not args.overwrite_output_dir and os.path.exists(fpath):
        file_mod_time = os.path.getmtime(fpath)
        current_time = time.time()
        n_days_in_seconds = N * 24 * 60 * 60

        if current_time - file_mod_time < n_days_in_seconds:
            logger.info(f"File {fpath} already exists. Skipping inference.")
            return True

    return False


def get_most_common_item(lst):
    frequencies = Counter(lst)
    most_common = frequencies.most_common(1)

    if len(most_common) > 0:
        # Return the most common item
        most_common_item = most_common[0][0]
    else:
        # Return the first item in the list
        most_common_item = lst[0]

    return most_common_item


def _get_all_prompts(args):
    """Utility function to retrieve all prompts."""
    question_prompts = VQA_PROMPT_COLLECTION[args.dataset_name]["question"]
    caption_prompts = VQA_PROMPT_COLLECTION[args.dataset_name]["caption"]
    if args.gen_model_name == "promptcap":
        caption_prompts = promptcap_prompts
        
    return (
        [f"{q},{c}" for q in question_prompts for c in caption_prompts]
        if args.vqa_format == "caption_vqa"
        else question_prompts
    )


def _cleanup():
    """Utility function to cleanup memory."""
    torch.cuda.empty_cache()  # Release unused GPU memory
    gc.collect()  # Trigger Python garbage collection
