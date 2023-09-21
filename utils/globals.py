# File name: globals.py
import os

# dataset config
DATASET_CONFIG = {
    "okvqa": {
        "val": {
            "question_file": "OpenEnded_mscoco_val2014_questions.json",
            "annotation_file": "mscoco_val2014_annotations.json",
            "image_root": "val2014/",
            "image_prefix": "COCO_val2014_",
        },
        "train": {
            "question_file": "OpenEnded_mscoco_train2014_questions.json",
            "annotation_file": "mscoco_train2014_annotations.json",
            "image_root": "train2014/",
            "image_prefix": "COCO_train2014_",
        },
    },
    "vqa_v2": {
        "val": {
            "question_file": "v2_OpenEnded_mscoco_val2014_questions.json",
            "annotation_file": "v2_mscoco_val2014_annotations.json",
            "image_root": "val2014/",
            "image_prefix": "COCO_val2014_",  # COCO_test2015_
        },
        "train": {
            "question_file": "v2_OpenEnded_mscoco_train2014_questions.json",
            "annotation_file": "v2_mscoco_train2014_annotations.json",
            "image_root": "train2014/",
            "image_prefix": "COCO_train2014_",
        },
    },
    "visual7w": {
        "image_root": "images/",
        "image_prefix": "v7w_",
    },
    "gqa": {
        "testdev_bal": {
            "annotation_file": "testdev_balanced_questions.json",  # this is eval set
            # "annotation_file": "testdev_small.json",
            "image_root": "images/",
            "image_prefix": "gqa",
        },
        "train_bal": {
            "annotation_file": "train_balanced_questions.json",
            "image_root": "images/",
            "image_prefix": "gqa",
        },
        "testdev_all": {
            "annotation_file": "testdev_all_questions.json",
            "image_root": "images/",
            "image_prefix": "gqa",
        },
    },
}

# Lavis models
LAVIS_MODELS = ["blip_vqa", "blip_caption", "blip2_t5_flant5xl", "blip2_t5_flant5xxl"]

# Information about the machine learning models
MODEL_CLS_INFO = {
    "lavis": {
        "blip_vqa": {"name": "blip_vqa", "model_type": "vqav2"},
        "blip_caption": {"name": "blip_caption", "model_type": "large_coco"},
        "blip2_flan_t5xl": {"name": "blip2_t5", "model_type": "pretrain_flant5xl"},
        "blip2_flant_5xxl": {"name": "blip2_t5", "model_type": "pretrain_flant5xxl"},
    },
    "hfformer": {
        "ofa_vqa": {"name": "OFA-Sys/ofa-huge-vqa"},
        "clip": {"name": "openai/clip-vit-base-patch32"},
        "git_base": {"name": "microsoft/git-base"},
        "git_large": {"name": "microsoft/git-large"},
        "git_large_textcaps": {"name": "microsoft/git-large-textcaps"},
        "blip2_opt27b": {"name": "Salesforce/blip2-opt-2.7b"},
        "blip2_opt67b": {"name": "Salesforce/blip2-opt-6.7b"},
        "blip2_flant5xl": {"name": "Salesforce/blip2-flan-t5-xl"},
        "blip2_flant5xxl": {"name": "Salesforce/blip2-flan-t5-xxl"},
        "flant5xl": {"name": "google/flan-t5-xl"},
        "flant5xxl": {"name": "google/flan-t5-xxl"},
        "opt27b": {"name": "facebook/opt-2.7b"},
        "opt67b": {"name": "facebook/opt-6.7b"},
        "kosmos2": {"name": "ydshieh/kosmos-2-patch14-224"},
    },
    "mlfoundations": {
        "open_flamingo_lamma": {
            "vision_encoder_path": "ViT-L-14",
            "name": "openflamingo/OpenFlamingo-9B",
            "lang_encoder_path": "luodian/llama-7b-hf",
            "tokenizer_path": "luodian/llama-7b-hf",
        }
    },
}


# Define an array of prompts.
okvqa_prompts = [
    "prefix_your_task_knowledge_qa_short_answer",
    # "prefix_your_task_knowledge_qa_short_answer_knn",
]
visual7w_prompts = [
    "prefix_your_task_grounded_qa_short_answer",
    # "prefix_your_task_grounded_qa_short_answer_knn",
]
gqa_prompts = [
    "prefix_your_task_compositional_qa_short_answer",
    # "prefix_your_task_compositional_qa_short_answer_knn",
]

vqa_v2_prompts = [
    "prefix_your_task_vqa_short_answer",
    # "prefix_your_task_vqa_short_answer_knn",
]

vqa_prompts = [
    "prefix_answer_the_following_question",
    "prefix_null",
    "prefix_question_answer",
    "prefix_question_short_answer",
    "prefix_think_step_by_step_rationale",
    "prefix_instruct_rationale",
]

vqa_fewshot_prompts = [
    "prefix_answer_the_following_question_knn",
    "prefix_question_answer_knn",
    "prefix_question_short_answer_knn",
]

caption_prompts = [
    "a_photo_of",
    "prefix_a_photo_of",
    "prefix_promptcap_a_photo_of",
]

VQA_PROMPT_COLLECTION = {
    "okvqa": {"caption": caption_prompts, "question": vqa_prompts + okvqa_prompts},
    "vqa_v2": {"caption": caption_prompts, "question": vqa_prompts + vqa_v2_prompts},
    "carets": {"caption": caption_prompts, "question": vqa_prompts},
    "visual7w": {"caption": caption_prompts, "question": vqa_prompts + visual7w_prompts},
    "aokvqa": {"caption": caption_prompts, "question": vqa_prompts + okvqa_prompts},
    "gqa": {"caption": caption_prompts, "question": vqa_prompts + gqa_prompts},
}

# grid search
num_beams_grid = [1, 2, 3, 4, 5, 6, 7, 8]
max_length_grid = [3, 5, 7, 10, 12, 15, 18, 20]
VQA_GRID_SEARCH = {"num_beams": num_beams_grid, "max_length": max_length_grid}
