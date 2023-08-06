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
parser.add_argument("--task_type", default="multiple_choice", type=str, help="Set the task type, e.g. `multiple_choice` or `open_ended`")
parser.add_argument(
    "--vqa_format",
    type=str,
    choices=["basic_qa", "caption_qa", "decompose_qa", "cot_qa"],
    choices=["basic_qa", "caption_qa", "decompose_qa", "cot_qa"],
)
parser.add_argument(
    "--decomposition_type",
    type=str,
    default=None,
    choices=["pipe", "cot", "dialogue"],
    help="Specify the type of question decomposition for use with the `decompose_qa` format. This parameter determines the method used, either Pipe, Chain-of-Thought or Dialogue.",
)
parser.add_argument(
    "--gen_model_name",
    type=str,
    default=None,
    choices=["gpt3", "chatgpt", "flant5", "promptcap", "blip2_opt27b", "blip2_opt67b", "blip2_t5_flant5xl", "blip2_t5_flant5xxl"],
    help="choose generator model type",
)
parser.add_argument(
    "--split",
    type=str,
    default="val",
    choices=["val", "testdev"],
    choices=["gpt3", "chatgpt", "gptj", "t5", "flant5", "opt", "git_large_textcaps", "promptcap"],
    help="choose generator model type",
)
parser.add_argument(
    "--split",
    type=str,
    default="val",
    choices=["val", "testdev"],
    help="choose generator model type",
)
parser.add_argument("--self_consistency", action="store_true", help="Set to run self consistency.")
parser.add_argument("--self_consistency", action="store_true", help="Set to run self consistency.")
parser.add_argument("--overwrite_output_dir", action="store_true", help="Set to True to overwrite the output file.")
parser.add_argument("--blind", action="store_true", help="Set to True to run blind where images are zeroed out.")
parser.add_argument("--batch_size", type=int, default=256, help="Set the batch size.")
parser.add_argument("--batch_size", type=int, default=128, help="Set the batch size.")
parser.add_argument("--num_workers", type=int, default=4, help="Set the number of workers.")

args = parser.parse_args()

if args.vqa_format == "decompose_qa" and args.decomposition_type is None:
    raise ValueError(
        "ERROR! decomposition_type is required for `decompose_qa` format. Please specify the decomposition_type."
    )
if args.vqa_format == "decompose_qa" and args.gen_model_name is None:
    raise ValueError("ERROR! gen_model_name is required for `decompose_qa` format. Please specify the gen_model_name.")

# cuda
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {args.device}")

for arg in vars(args):
    logger.info(f"{arg}: {getattr(args, arg)}")

# Find the model class and the corresponding model type
model_class, model_info = None, None
for cls, models in MODEL_CLS_INFO.items():
    if args.model_name in models:
        model_class = cls
        model_info = models[args.model_name]
        break

if model_class is None or model_info is None:
    raise ValueError(f"ERROR! Unsupported model_name: {args.model_name}")

# Generate the inference_function_map dictionary based on the MODEL_CLS_INFO dictionary
inference_fn_suffix_map = {
    "winoground": "winoground",
    "carets": "vqa",
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
            inference_function_map[(cls, model_name, args.dataset_name)] = f"hfformer.inference_{fn_suffix}"
        elif cls == "mlfoundations":
            inference_function_map[(cls, model_name, args.dataset_name)] = f"mlfoundations.inference_{fn_suffix}"

try:
    module_name = inference_function_map[(model_class, args.model_name, args.dataset_name)]
    inference_module = __import__(module_name, fromlist=["run_inference"])
    inference_module.run_inference(args)
except KeyError:
    raise ValueError(
        f"ERROR! Unsupported combination of model_class ({model_class}), model_name ({args.model_name}), and dataset_name ({args.dataset_name})"
    )