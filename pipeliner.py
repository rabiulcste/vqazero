import argparse
import os

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from utils.logger import Logger
from utils.prompt_engineering import (
    DecomposeGeneratorForWinoground,
    QuestionGeneratorforWinoground,
    CaptionGeneratorforWinoground,
)
from utils.handler import PromptingHandler
from utils.globals import MODEL_CLS_INFO

VALID_MODELS = list(MODEL_CLS_INFO["lavis"].keys()) + list(
    MODEL_CLS_INFO["hfformer"].keys()
)

logger = Logger(__name__)

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Device: {device}")


def generate_questions_for_winoground_dataset(args):
    """
    Generate questions for Winogrande dataset and save output to "output_dir"
    """
    model_name = args.gen_model_name
    qgen = QuestionGeneratorforWinoground(model_name)

    # setup prompt
    prompt_name = args.prompt_name

    handler = PromptingHandler(
        args.dataset_name, prompt_name, subset_name="api_question"
    )
    template_expr = handler.prompt.jinja if handler.prompt else ""
    logger.info(f"PROMPT TEMPLATE: {template_expr}")

    # Generate questions for Winogrande dataset and save output to "output_dir"
    winoground = load_dataset("facebook/winoground", use_auth_token=True)[
        "test"
    ]  # winoground-dataset

    output_dir = f"output/gpt3-api/generated-questions/{prompt_name}"
    logger.info(f"Output directory is set to {output_dir}")

    os.makedirs(output_dir, exist_ok=True)
    qgen.generate_questions(winoground, handler, output_dir)

    # access the generated questions
    example_id = np.random.randint(len(winoground))
    generated_questions = qgen.get_generated_questions_for_winoground(
        example_id, output_dir
    )

    # print the generated questions for the first example in the dataset
    example_id = generated_questions["id"]
    caption_0_question = generated_questions["caption_0"].strip()
    caption_1_question = generated_questions["caption_1"].strip()
    logger.info(f"Generated questions for example {example_id}:")
    logger.info(f"Caption 0 question: {caption_0_question}")
    logger.info(f"Caption 1 question: {caption_1_question}")


# chatgpt based subquestions generation
def generate_subquestions_for_winoground_dataset(args):
    """
    Generate subquestions for Winogrande dataset and save output to "output_dir"
    """
    decompose_gen = DecomposeGeneratorForWinoground(
        dataset_name=args.dataset_name,
        gen_model_name=args.gen_model_name,
        model_name=args.model_name,
    )

    # setup prompt
    prompt_name = args.prompt_name

    handler = PromptingHandler(
        args.dataset_name, prompt_name, subset_name="api_decomposition"
    )
    template_expr = handler.prompt.jinja if handler.prompt else ""
    logger.info(f"PROMPT TEMPLATE: {template_expr}")

    # Generate questions for Winogrande dataset and save output to "output_dir"
    winoground_dataset = load_dataset("facebook/winoground", use_auth_token=True)[
        "test"
    ]  # winoground-dataset

    decompose_gen.generate_decomposed_subquestions(winoground_dataset, handler)
    decompose_gen.print_sample_generated_questions(prompt_name)


def generate_captions_for_winoground_dataset(args):
    """
    Generate captions for Winogrande dataset and save output to "output_dir"
    """
    model_name = args.gen_model_name
    c_gen = CaptionGeneratorforWinoground(model_name, device)
    c_gen.generate_captions()

    # access the generated questions
    example_id = np.random.randint(len(c_gen.dataset))
    generated_questions = c_gen.get_generated_captions(example_id)

    # print the generated questions for the first example in the dataset
    example_id = generated_questions["id"]
    caption_0_question = generated_questions["caption_0"].strip()
    caption_1_question = generated_questions["caption_1"].strip()
    logger.info(f"Generated captions for example {example_id}:")
    logger.info(f"Caption 0: {caption_0_question}")
    logger.info(f"Caption 1: {caption_1_question}")


def generate_rationale_for_winoground_dataset(args):
    """
    Generate rationale for Winogrande dataset and save output to "output_dir"
    """
    decompose_gen = DecomposeGeneratorForWinoground(
        args.dataset_name,
        args.gen_model_name,
        args.model_name,
    )
    decompose_gen.generate_rationale(args.prompt_name, args.decomposition_type)


def generate_eval_for_winoground_dataset(args):
    """
    Generate eval for Winogrande dataset and save output to "output_dir"
    """
    decompose_gen = DecomposeGeneratorForWinoground(
        args.dataset_name,
        args.gen_model_name,
        args.model_name,
    )
    decompose_gen.generate_eval(args.prompt_name, args.decomposition_type)


if __name__ == "__main__":
    # Define command-line arguments
    parser = argparse.ArgumentParser(description="Generators for Winoground dataset")
    parser.add_argument("--dataset_name", type=str, default="winoground", required=True)
    parser.add_argument(
        "--gen_model_name",
        type=str,
        choices=["gpt3", "chatgpt", "gptj", "t5", "flant5", "opt", "blip_caption"],
        required=True,
        help="choose generator model type",
    )
    parser.add_argument(
        "--model_name",
        default="blip_vqa",
        type=str,
        help="provide VQA or VL model",
        choices=VALID_MODELS,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/generated-questions",
        help="output directory",
    )
    parser.add_argument(
        "--prompt_name",
        type=str,
        default="convert_question_to_binary_vqa",
        help="prompt name",
        required=False,
    )
    parser.add_argument(
        "--decomposition_type",
        type=str,
        default=None,
        choices=["pipe", "cot", "dialogue"],
        help="Specify the type of question decomposition for use with the `decompose_qa` format. This parameter determines the method used, either Pipe, Chain-of-Thought or Dialogue.",
    )
    parser.add_argument(
        "--function",
        type=str,
        default="generate_questions",
        help="function to run",
        required=True,
        choices=[
            "generate_questions",
            "generate_captions",
            "generate_subquestions",
            "generate_rationale",
            "generate_eval"
        ],
    )

    args = parser.parse_args()
    logger.info(vars(args))

    if args.function == "generate_captions":
        generate_captions_for_winoground_dataset(args)
    elif args.function == "generate_questions":
        generate_questions_for_winoground_dataset(args)
    elif args.function == "generate_subquestions":
        generate_subquestions_for_winoground_dataset(args)
    elif args.function == "generate_rationale":
        logger.warning(
            f'"args.decompositon_type" is set to {args.decomposition_type}.'
        )
        generate_rationale_for_winoground_dataset(args)
    elif args.function == "generate_eval":
        generate_eval_for_winoground_dataset(args)
    else:
        raise NotImplementedError(
            "Funtion not implemented yet but provided in args {args.function}"
        )
