# script supports following datasets: OKVQA, AOKVQA, GQA, Visual7W and VQAv2

import json
import os
from typing import List

import torch
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForVision2Seq,
    AutoProcessor,
    AutoTokenizer,
    Blip2ForConditionalGeneration,
    Blip2Processor,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

from dataset_zoo.custom_dataset import VQADataset, collate_fn_builder
from dataset_zoo.nearest_neighbor import cache_nearest_neighbor_data
from evals.answer_postprocess import answer_postprocess_batch
from evals.answer_postprocess_vicuna_llm import AnswerPostProcessLLM
from evals.vicuna_llm_evals import extract_answers_from_predictions_vicunallm
from evals.vqa_accuracy import eval_aokvqa, eval_gqa, eval_visual7w, eval_vqa
from utils.config import HUGGINFACE_HUB_DIR
from utils.globals import MODEL_CLS_INFO
from utils.helpers import _cleanup, _get_all_prompts, update_configs
from utils.logger import Logger
from vqa_zero.caption_generation import CaptionGenerator, PromptCapGenerarator
from vqa_zero.inference_utils import (
    get_output_dir_path,
    get_prompt_template_handler,
    save_to_json,
    save_vqa_answers,
    save_vqa_answers_chunked,
    set_seed,
)
from vqa_zero.rationale_generation import ChainOfThoughtGenerator

set_seed(42)

logger = Logger(__name__)


def save_and_evaluate_predictions(args, output_dir, predictions, vicuna_ans_parser=False):
    multiple_choice = args.task_type == "multiple_choice"
    dataset_eval_mapping = {
        "okvqa": (save_vqa_answers, eval_vqa),
        "vqa_v2": (save_vqa_answers, eval_vqa),  # eval_vqa might not work for vqa_v2 because of chunking
        "gqa": (None, eval_gqa),
        "visual7w": (
            None,
            lambda args, dir, parser: eval_visual7w(
                args, dir, predictions, multiple_choice=multiple_choice, vicuna_ans_parser=parser
            ),
        ),
        "aokvqa": (
            None,
            lambda args, dir, parser: eval_aokvqa(
                args, dir, predictions, multiple_choice=multiple_choice, vicuna_ans_parser=parser
            ),
        ),
    }

    save_func, eval_func = dataset_eval_mapping.get(args.dataset_name, (None, None))

    if save_func:
        save_func(output_dir, args.dataset_name, predictions, args.vicuna_ans_parser)

    if eval_func:
        eval_func(args, output_dir, vicuna_ans_parser)


def handle_caption_and_cot_generation(args, prompt_handler):
    if args.gen_model_name is None:
        args.gen_model_name = args.model_name

    def get_caption_generator(args):
        if args.gen_model_name == "promptcap":
            return PromptCapGenerarator(args, device="cuda")
        else:
            return CaptionGenerator(args, device="cuda")

    if args.vqa_format == "caption_vqa":
        caption_gen = get_caption_generator(args)
        if args.few_shot:
            caption_gen.generate_caption(prompt_handler, "train")
        caption_gen.generate_caption(prompt_handler, args.split)

    elif args.vqa_format == "cot_vqa":
        cot_gen = ChainOfThoughtGenerator(args, device="cuda")
        if args.few_shot:
            cot_gen.generate_chain_of_thought_rationale(prompt_handler, "train")

        if "iterative" in args.prompt_name or "mixer" in args.prompt_name:
            cot_gen.generate_chain_of_thought_rationale(prompt_handler, args.split)


def perform_vqa_inference(args):
    if args.vqa_format not in ["standard_vqa", "caption_vqa", "cot_vqa"]:
        raise NotImplementedError(
            f"Provided VQA format {args.vqa_format} is either not implemented yet or invalid argument provided."
        )
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.few_shot:
        cache_nearest_neighbor_data(args.dataset_name, args.task_type == "multiple_choice")

    # get prompting handler
    prompt_handler, template_expr = get_prompt_template_handler(args)

    if isinstance(prompt_handler, List):
        prompt_handler, context_prompt_handler = prompt_handler
        handle_caption_and_cot_generation(args, context_prompt_handler)

    model_name = MODEL_CLS_INFO["hfformer"][args.model_name]["name"]

    processor, tokenizer = None, None
    if args.model_name in ["flant5xl", "flant5xxl"]:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        model = T5ForConditionalGeneration.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

    elif args.model_name in ["opt27b", "opt67b"]:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
        tokenizer.padding_side = "left"

    elif args.model_name == "kosmos2":
        model = AutoModelForVision2Seq.from_pretrained(model_name, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"
        model = model.to(device)

    elif args.model_name in ["open_flamingo_lamma", "open_flamingo_mpt"]:
        from open_flamingo import create_model_and_transforms

        model, image_processor, tokenizer = create_model_and_transforms(
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path="anas-awadalla/mpt-1b-redpajama-200b",
            tokenizer_path="anas-awadalla/mpt-1b-redpajama-200b",
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
        processor = FlamingoProcessor(tokenizer, image_processor, device, torch.bfloat16)

    elif args.model_name == "llava":
        from LLaVA.llava.conversation import SeparatorStyle, conv_templates
        from LLaVA.llava.model.builder import load_pretrained_model
        from LLaVA.llava.utils import disable_torch_init
        from LLaVA.llava.mm_utils import get_model_name_from_path, KeywordsStoppingCriteria
        from dataset_zoo.custom_processor import LlaVaProcessor

        disable_torch_init()
        model_path = os.path.expanduser(model_name)
        model_name = get_model_name_from_path(model_path)
        model_name = get_model_name_from_path(model_path)
        model_base = MODEL_CLS_INFO["hfformer"][args.model_name]["base"]
        tokenizer_, model, image_processor, context_len = load_pretrained_model(model_path, model_base, model_name)
        processor = LlaVaProcessor(tokenizer_, image_processor, model.config.mm_use_im_start_end)

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
        "prompt_handler": prompt_handler,
        "model_name": args.model_name,
        "split": args.split,
    }
    dataset = VQADataset(**dataset_args)
    collate_fn = collate_fn_builder(processor, tokenizer)
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
        prompt = batch["prompted_question"]
        if args.blind:  # zero out each images pixels, note that images are list of images
            images = [TF.to_tensor(img) * 0 for img in images]

        logger.debug(f"TEXT INPUT: {json.dumps(prompt, indent=2)}")

        if args.model_name in ["flant5xl", "flant5xxl", "opt27b", "opt67b"]:
            generated_ids = model.generate(
                input_ids=batch["input_ids"].to("cuda"),
                attention_mask=batch["attention_mask"].to("cuda"),
                max_new_tokens=args.max_length,
                num_beams=args.num_beams,
                length_penalty=args.length_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
            )
            generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            if processor is not None and processor.padding_side == "left":
                generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
                generated_outputs = [processor.post_process_generation(gt) for gt in generated_outputs]

        elif args.model_name == "kosmos2":
            generated_ids = model.generate(
                pixel_values=batch["pixel_values"].to("cuda"),
                input_ids=batch["input_ids"][:, :-1].to("cuda"),
                attention_mask=batch["attention_mask"][:, :-1].to("cuda"),
                img_features=None,
                img_attn_mask=batch["img_attn_mask"][:, :-1].to("cuda"),
                max_new_tokens=args.max_length,
                length_penalty=args.length_penalty,
                num_beams=args.num_beams,
            )
            generated_ids = generated_ids[:, batch["input_ids"].shape[1] - 1 :]
            generated_texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            generated_outputs, entities_list = zip(*[processor.post_process_generation(gt) for gt in generated_texts])

        elif args.model_name == "open_flamingo_mpt":
            # TODO: torch.bfloat16 `dtype` is not used here
            batch_images = batch["image_tensors"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                generated_ids = model.generate(
                    vision_x=batch_images,
                    lang_x=input_ids,
                    attention_mask=attention_mask,
                    min_new_tokens=0,
                    max_new_tokens=args.max_length,
                    num_beams=3,
                    length_penalty=args.length_penalty,
                )
            generated_ids = generated_ids[:, batch["input_ids"].shape[1] :]
            generated_outputs = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        elif args.model_name == "llava":
            conv = conv_templates[processor.conv_mode].copy()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = (
                [KeywordsStoppingCriteria(keywords, tokenizer_, input_ids)] if conv.version == "v0" else None
            )

            generated_outputs = []
            for i in range(args.batch_size):  # doesn't work with batch_size > 1
                input_ids = batch["input_ids"][i]
                image_tensor = batch["image_tensors"][i]
                input_ids = input_ids.cuda()
                with torch.inference_mode():
                    output_ids = model.generate(
                        input_ids,
                        images=image_tensor.unsqueeze(0).half().cuda(),
                        do_sample=False,
                        temperature=0.2,
                        max_new_tokens=args.max_length,
                        length_penalty=args.length_penalty,
                        use_cache=True,
                        stopping_criteria=stopping_criteria,
                    )

                input_token_len = input_ids.shape[1]
                n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
                if n_diff_input_output > 0:
                    logger.warning(f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids")
                outputs = tokenizer_.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
                outputs = outputs.strip()
                if outputs.endswith(stop_str):
                    outputs = outputs[: -len(stop_str)]
                outputs = outputs.strip()
                generated_outputs.append(outputs)

        else:
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
            generated_outputs = processor.batch_decode(generated_ids, skip_special_tokens=True)

        keys = ["question", "prompted_question", "question_id", "answer"]
        optional_keys = ["reasoning_paths"]
        for i, output in enumerate(generated_outputs):
            example = {key: batch[key][i] for key in keys}
            example.update(
                {key: batch[key][i] for key in optional_keys if key in batch}
            )  # Add optional keys if they exist in batch
            example["generated_output"] = output
            predictions[example["question_id"]] = example

        formatted_message_items = [
            f"question = {question}, prediction = {pred}"
            for question, pred in zip(batch["question"][-5:], generated_outputs[-5:])
        ]
        formatted_message = "\n".join(formatted_message_items)
        logger.info(formatted_message)

    return predictions


def evaluate_predictions(args, predictions, **kwargs):
    logger.info("Running evaluation...")
    qids = list(predictions.keys())
    batch_data = [predictions[qid] for qid in qids]
    batch_data = answer_postprocess_batch(args, batch_data, logger)
    output_dir = get_output_dir_path(args, **kwargs)

    save_to_json(os.path.join(output_dir, "predictions.json"), predictions)  # for debugging
    save_to_json(os.path.join(output_dir, "configs.json"), vars(args))
    save_and_evaluate_predictions(args, output_dir, predictions)


def evaluate_predictions_vicuna(args, **kwargs):
    logger.info("Running evaluation (vicunallm)...")
    output_dir = get_output_dir_path(args, **kwargs)
    fpath = os.path.join(output_dir, "predictions.json")
    with open(fpath, "r") as f:
        predictions = json.load(f)
    answer_extractor = kwargs.get("answer_extractor")
    predictions = extract_answers_from_predictions_vicunallm(args, predictions, answer_extractor, batch_size=64)
    save_to_json(fpath, predictions)  # for debugging
    save_and_evaluate_predictions(args, output_dir, predictions, vicuna_ans_parser=True)


def run_inference(args):
    """
    Performs Visual Question Answering (VQA) inference using the BLIP2 model varinats.
    The function supports three different formats for the VQA task:
        - standard vqa: the model is given an image and a question and it has to answer the question.
        - caption vqa: the model is given an image, a question, and a caption. The model has to answer the question,
          with the caption providing context for improving the answer.
        - chain-of-thought vqa: the model is given an image, a question. The model has to answer the question and provide reasoning.
    """
    all_prompts = [args.prompt_name]
    # in case, you want to evaluate all prompts in the collection
    if not args.prompt_name:  # if prompt name is not provided, we look for a list of prompts in the globals.py file
        all_prompts = _get_all_prompts(args)

    logger.info(f"Total prompts: {len(all_prompts)} will be evaluated.")

    for prompt_name in all_prompts:
        args.prompt_name = prompt_name
        logger.info(f'Selected prompt name :"{args.prompt_name}"')
        fpath = os.path.join(get_output_dir_path(args), "result_meta.json")
        if not args.overwrite_output_dir and os.path.exists(fpath):
            logger.info(f"File {fpath} already exists. Skipping inference.")
            continue

        args = update_configs(args)
        predictions = perform_vqa_inference(args)
        evaluate_predictions(args, predictions)
        args = update_configs(args)

        _cleanup()

    if args.vicuna_ans_parser:
        language_model_name = "lmsys/vicuna-7b-v1.5"
        answer_extractor = AnswerPostProcessLLM(language_model_name=language_model_name, device="cuda")
        answer_extractor.load_pretrained_model_tokenizer()
        ctx_model_name = "opt" if "opt" in args.model_name else "llava"
        answer_extractor._load_in_context_examples(args.dataset_name, ctx_model_name)

        for prompt_name in all_prompts:
            args.prompt_name = prompt_name
            fpath = os.path.join(get_output_dir_path(args), "result_meta_vicuna.json")
            if not args.overwrite_output_dir and os.path.exists(fpath):
                logger.info(f"File {fpath} already exists. Skipping inference.")
                continue

            evaluate_predictions_vicuna(args, answer_extractor=answer_extractor)
