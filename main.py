import argparse

import torch

from utils.globals import MODEL_CLS_INFO
from utils.logger import Logger

logger = Logger(__name__)

parser = argparse.ArgumentParser(description="main")
parser.add_argument("--model_name", type=str, required=True, help="Set the model name, e.g. `blip_vqa`")
parser.add_argument("--dataset_name", type=str, help="Set the dataset name, e.g. `winoground`", required=True)
parser.add_argument("--grid_search", action="store_true", help="Set to True to run grid search.")
parser.add_argument("--prompt_name", type=str, help="Set the prompt name, e.g. `does_caption_match_wo_answer`")
parser.add_argument(
    "--task_type",
    default="open_ended",
    choices=["multiple_choice", "open_ended"],
    type=str,
    help="Set the task type, e.g. `multiple_choice` or `open_ended`",
)
parser.add_argument(
    "--vqa_format",
    type=str,
    choices=["standard_vqa", "caption_vqa", "cot_vqa"],
)
parser.add_argument(
    "--gen_model_name",
    type=str,
    default=None,
    choices=[
        "gpt3",
        "chatgpt",
        "flan_t5",
        "promptcap",
        "blip2_opt27b",
        "blip2_opt67b",
        "blip2_flant5xl",
        "blip2_flant5xxl",
        "kosmos2",
    ],
    help="choose generator model type",
)
parser.add_argument(
    "--split_name",
    type=str,
    default="val",
    choices=["val", "testdev_bal"],
    help="choose generator model type",
)
parser.add_argument("--self_consistency", action="store_true", help="Set to run self consistency.")
parser.add_argument("--few_shot", action="store_true", help="Set to run self consistency.")
parser.add_argument("--nearest_neighbor_threshold", type=float, default=None, help="Set the nearest neighbor threshold.")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Set to True to overwrite the output file.")
parser.add_argument("--blind", action="store_true", help="Set to True to run blind where images are zeroed out.")
parser.add_argument("--vicuna_ans_parser", action="store_true", help="Set to True to run vicuna eval.")
parser.add_argument("--batch_size", type=int, default=32, help="Set the batch size.")
parser.add_argument("--num_workers", type=int, default=4, help="Set the number of workers.")
parser.add_argument("--chunk_id", type=int, default=None, help="Set the chunk id.")

args = parser.parse_args()

# cuda
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {args.device}")

for arg in vars(args):
    logger.info(f"{arg}: {getattr(args, arg)}")

# Generate the inference_function_map dictionary based on the MODEL_CLS_INFO dictionary
inference_fn_suffix_map = {
    "winoground": "winoground_vqa",
    "okvqa": "vqa",
    "vqa_v2": "vqa",
    "visual7w": "vqa",
    "aokvqa": "vqa",
    "gqa": "vqa",
}

inference_function_map = {}
for cls, models in MODEL_CLS_INFO.items():
    fn_suffix = inference_fn_suffix_map[args.dataset_name]
    for model_name in models:
        if cls == "lavis":
            inference_function_map[(cls, model_name, args.dataset_name)] = f"lavis_blip.inference_{fn_suffix}"
        elif cls == "hfformer":
            inference_function_map[(cls, model_name, args.dataset_name)] = f"vqa_zero.inference_{fn_suffix}"


# Find the model class and the corresponding model type
model_class, model_info = None, None
for cls, models in MODEL_CLS_INFO.items():
    if args.model_name in models:
        model_class = cls
        model_info = models[args.model_name]
        break

if "llava" in args.model_name or "minigpt4" in args.model_name: # custom checkpoints
    model_class = "hfformer"
    fn_suffix = inference_fn_suffix_map[args.dataset_name]
    inference_function_map[(model_class, args.model_name, args.dataset_name)] = f"vqa_zero.inference_{fn_suffix}"


if model_class is None and model_info is None:
    raise ValueError(f"ERROR! Unsupported model_name: {args.model_name}")


try:
    module_name = inference_function_map[(model_class, args.model_name, args.dataset_name)]
    inference_module = __import__(module_name, fromlist=["run_inference"])
    inference_module.main(args)
except KeyError:
    raise ValueError(
        f"ERROR! Unsupported combination of model_class ({model_class}), model_name ({args.model_name}), and dataset_name ({args.dataset_name})"
    )
