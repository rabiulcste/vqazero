import gc
import json
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch

from utils.globals import VQA_PROMPT_COLLECTION, promptcap_prompts
from utils.logger import Logger

logger = Logger(__name__)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle numpy numbers
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        # Handle pandas DataFrame and Series
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, pd.Series):
            return obj.to_dict()

        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        if hasattr(obj, "tolist"):  # Convert tensors to lists
            return obj.tolist()
        elif hasattr(obj, "name"):  # Convert PyTorch device to its name string
            return obj.name
        elif isinstance(obj, type):  # If it's a type/class object
            return str(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, torch.device):  # Handling torch.device objects
            return str(obj)
        elif isinstance(obj, torch.dtype):  # Handling torch.dtype objects
            return str(obj)

        # Handle other non-serializable objects or custom classes
        # By default, convert them to string (change this if needed)
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


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
