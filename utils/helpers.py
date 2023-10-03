import gc
import json

import torch

from utils.globals import VQA_PROMPT_COLLECTION, promptcap_prompts
from utils.logger import Logger

logger = Logger(__name__)


class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
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
        # You can add more custom handlers here if needed
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
