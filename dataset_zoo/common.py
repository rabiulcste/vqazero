import json
import os
import random
from collections import Counter, defaultdict
from typing import Dict, List, Union

from tqdm import tqdm

from utils.config import COCO_DATASET_DIR, VQA_DATASET_DIR
from utils.globals import DATASET_CONFIG
from utils.logger import Logger

logger = Logger(__name__)

random.seed(42)


def get_vqa_dataset(dataset_name: str, split_name: str, multiple_choice: bool = False, chunk_id=None) -> List[Dict]:
    if dataset_name == "aokvqa":
        dataset = load_aokvqa_dataset_from_json(
            dataset_name, split_name, multiple_choice=multiple_choice, chunk_id=chunk_id
        )
    elif dataset_name == "visual7w":
        dataset = load_visual7w_dataset_from_json(dataset_name, split_name, multiple_choice=multiple_choice)
    elif dataset_name in ["vqa_v2", "okvqa"]:
        dataset = load_vqa_dataset_from_json(dataset_name, split_name, chunk_id=chunk_id)
    elif dataset_name == "gqa":
        dataset = load_gqa_dataset_from_json(dataset_name, split_name)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented yet.")

    return dataset


def get_coco_path_image_id(coco_dir: str, split_name: str, image_id):
    return os.path.join(coco_dir, f"{split_name}2017", f"{image_id:012}.jpg")


def get_chunked_question_ids(dataset_dir: str, split_name: str, chunk_id: Union[int, None] = None):
    chunked_question_ids = []
    if chunk_id is not None:
        chunked_question_ids_fpath = os.path.join(
            dataset_dir, "extras", "chunked", f"chunked_question_ids_{split_name}.json"
        )
        with open(chunked_question_ids_fpath) as f:
            chunked_data = json.load(f)
        chunked_question_ids = chunked_data[chunk_id]
        logger.info(f"Loading chunk {chunk_id} of {len(chunked_data)} from {split_name} split_name")

    return chunked_question_ids


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


def load_aokvqa_dataset_from_json(
    dataset_name: str, split_name: str, multiple_choice: bool, version="v1p0", chunk_id=None
) -> List[Dict]:
    """
    Builds a dataset from a JSON file using the AOKVQA dataset format.
    Returns:
        A list of dictionaries, where each dictionary represents a QA object.
        Each QA object has the following keys: 'question_id', 'question', 'image_path', and 'direct_answers'.
    """
    assert split_name in ["train", "val", "test", "test_w_ans"]
    dataset_dir = os.path.join(VQA_DATASET_DIR, dataset_name)
    annotated_data = json.load(open(os.path.join(dataset_dir, f"aokvqa_{version}_{split_name}.json")))
    coco_dir = os.path.join(COCO_DATASET_DIR, "images")

    chunked_question_ids = get_chunked_question_ids(dataset_dir, split_name, chunk_id)

    dataset = []
    for qa_obj in tqdm(annotated_data, desc=f"Preprocessing {dataset_name} dataset"):
        question_id = qa_obj["question_id"]
        if chunk_id is not None and question_id not in chunked_question_ids:
            continue

        if multiple_choice:
            answer = qa_obj["choices"][qa_obj["correct_choice_idx"]]
        else:
            answer = get_most_common_item(qa_obj["direct_answers"])  # hacky way to get the answer

        choices = qa_obj["choices"]
        random.shuffle(choices)
        image_path = get_coco_path_image_id(coco_dir, split_name, qa_obj["image_id"])
        qa_dict = {
            "image_id": qa_obj["image_id"],
            "question_id": question_id,
            "question": qa_obj["question"],
            "image_path": image_path,
            "answer": answer,
            "answers": qa_obj["direct_answers"],
        }
        if multiple_choice:
            qa_dict["choice"] = choices
        dataset.append(qa_dict)

    logger.info(json.dumps(dataset[0], indent=4))

    return dataset


def load_vqa_dataset_from_json(dataset_name: str, split_name: str, chunk_id=None) -> List[Dict]:
    dataset_dir = os.path.join(VQA_DATASET_DIR, dataset_name)
    question_file = DATASET_CONFIG[dataset_name][split_name]["question_file"]
    annotation_file = DATASET_CONFIG[dataset_name][split_name]["annotation_file"]
    image_dir = DATASET_CONFIG[dataset_name][split_name]["image_root"]
    coco_prefix = DATASET_CONFIG[dataset_name][split_name]["image_prefix"]

    question_data_fpath = os.path.join(dataset_dir, question_file)
    annotation_data_fpath = os.path.join(dataset_dir, annotation_file)
    image_root = os.path.join(COCO_DATASET_DIR, "images", image_dir)

    chunked_question_ids = get_chunked_question_ids(dataset_dir, split_name, chunk_id)

    with open(question_data_fpath) as f:
        question_data = json.load(f)

    with open(annotation_data_fpath, encoding="utf-8") as f:
        answer_data = json.load(f)
        answer_data = answer_data["annotations"]

    answer_data_dict = {}
    for answer_obj in answer_data:
        question_id = answer_obj["question_id"]
        answer_data_dict[question_id] = answer_obj

    dataset = []
    for qa_obj in tqdm(question_data["questions"], desc=f"Preprocessing {dataset_name} dataset"):
        question_id = qa_obj["question_id"]
        if chunk_id is not None and question_id not in chunked_question_ids:
            continue
        image_id = qa_obj["image_id"]
        image_path = os.path.join(image_root, coco_prefix + "{:012d}".format(image_id) + ".jpg")
        qa_obj.update({"image_path": image_path})  # adding image path to the question object

        # adding answer to the question object
        answer_data = answer_data_dict[question_id]
        qa_obj.update(answer_data)
        dataset.append(qa_obj)

    logger.info(json.dumps(dataset[0], indent=4))

    return dataset


def load_gqa_dataset_from_json(dataset_name: str, split_name: str) -> List[Dict]:
    annotation_file = DATASET_CONFIG[dataset_name][split_name]["annotation_file"]
    image_dir = DATASET_CONFIG[dataset_name][split_name]["image_root"]

    annotation_data_fpath = os.path.join(VQA_DATASET_DIR, dataset_name, annotation_file)
    image_root = os.path.join(VQA_DATASET_DIR, dataset_name, image_dir)

    with open(annotation_data_fpath) as f:
        annotated_data = json.load(f)

    dataset = []
    for idx in annotated_data:
        qa_obj = annotated_data[idx]
        image_path = os.path.join(image_root, qa_obj["imageId"] + ".jpg")
        data = {
            "image_path": image_path,
            "question": qa_obj["question"],
            "answer": qa_obj["answer"],
            "question_id": idx,
            "image_id": qa_obj["imageId"],
        }
        dataset.append(data)

    logger.info(json.dumps(dataset[0], indent=4))

    return dataset


def load_visual7w_dataset_from_json(dataset_name: str, split_name: str, multiple_choice: bool = True) -> List[Dict]:
    fname = os.path.join(VQA_DATASET_DIR, dataset_name, "dataset.json")
    dataset = json.load(open(fname, "r"))
    dataset_split_name = defaultdict(list)
    for img in dataset["images"]:
        dataset_split_name[img["split"]].append(img)

    image_dir = DATASET_CONFIG[dataset_name]["image_root"]
    img_prefix = DATASET_CONFIG[dataset_name]["image_prefix"]
    image_root = os.path.join(VQA_DATASET_DIR, dataset_name, image_dir)

    dataset = []
    for img in tqdm(dataset_split_name[split_name], desc=f"Preprocessing {dataset_name} dataset"):
        for qa_obj in img["qa_pairs"]:
            dataset.append(
                {
                    "image_path": os.path.join(image_root, f"{img_prefix}{qa_obj['image_id']}.jpg"),
                    "image_id": qa_obj["image_id"],
                    "question": qa_obj["question"],
                    "choice": qa_obj["multiple_choices"] + [qa_obj["answer"]],
                    "question_id": qa_obj["qa_id"],
                    "answer": qa_obj["answer"],
                }
            )

    if split_name == "train":
        random.shuffle(dataset)
        dataset = dataset[:5000]
        logger.warning("WARNING: Truncating train dataset to 5000 samples")

    logger.info(json.dumps(dataset[0], indent=4))

    return dataset
