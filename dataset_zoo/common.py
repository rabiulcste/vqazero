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


def get_vqa_dataset(dataset_name: str, split: str, multiple_choice: bool = False, chunk_id=None) -> List[Dict]:
    if dataset_name == "aokvqa":
        dataset = load_aokvqa_dataset_from_json(dataset_name, split, multiple_choice=multiple_choice, chunk_id=chunk_id)
    elif dataset_name == "visual7w":
        dataset = load_visual7w_dataset_from_json(dataset_name, split, multiple_choice=multiple_choice)
    elif dataset_name in ["vqa_v2", "okvqa"]:
        dataset = load_vqa_dataset_from_json(dataset_name, split, chunk_id=chunk_id)
    elif dataset_name == "gqa":
        dataset = load_gqa_dataset_from_json(dataset_name, split)
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not implemented yet.")

    return dataset


def get_coco_path_image_id(coco_dir: str, split: str, image_id):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def get_chunked_question_ids(dataset_dir: str, split: str, chunk_id: Union[int, None] = None):
    chunked_question_ids = []
    if chunk_id is not None:
        chunked_question_ids_fpath = os.path.join(dataset_dir, f"chunked_question_ids_{split}.json")
        with open(chunked_question_ids_fpath) as f:
            chunked_data = json.load(f)
        chunked_question_ids = chunked_data[chunk_id]
        logger.info(f"Loading chunk {chunk_id} of {len(chunked_data)} from {split} split")

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
    dataset_name: str, split: str, multiple_choice: bool, version="v1p0", chunk_id=None
) -> List[Dict]:
    """
    Builds a dataset from a JSON file using the AOKVQA dataset format.
    Returns:
        A list of dictionaries, where each dictionary represents a QA object.
        Each QA object has the following keys: 'question_id', 'question', 'image_path', and 'direct_answers'.
    """
    assert split in ["train", "val", "test", "test_w_ans"]
    dataset_dir = os.path.join(VQA_DATASET_DIR, "datasets", dataset_name)
    dataset = json.load(open(os.path.join(dataset_dir, f"aokvqa_{version}_{split}.json")))
    coco_dir = os.path.join(COCO_DATASET_DIR, "images")

    chunked_question_ids = get_chunked_question_ids(dataset_dir, split, chunk_id)

    qa_objects = []
    for qa_obj in tqdm(dataset, desc=f"Preprocessing {dataset_name} dataset"):
        question_id = qa_obj["question_id"]
        if chunk_id is not None and question_id not in chunked_question_ids:
            continue

        image_id = qa_obj["image_id"]
        question = qa_obj["question"]
        choices = qa_obj["choices"]
        if multiple_choice:
            answer = qa_obj["choices"][qa_obj["correct_choice_idx"]]
        else:
            answer = get_most_common_item(qa_obj["direct_answers"])  # hacky way to get the answer
        random.shuffle(choices)
        image_path = get_coco_path_image_id(coco_dir, split, image_id)
        qa_dict = {
            "image_id": image_id,
            "question_id": question_id,
            "question": question,
            "image_path": image_path,
            "answer": answer,
        }
        if multiple_choice:
            qa_dict["choice"] = choices

        qa_objects.append(qa_dict)

    logger.info(json.dumps(qa_objects[0], indent=4))

    return qa_objects


def load_vqa_dataset_from_json(dataset_name: str, split: str, chunk_id=None) -> List[Dict]:
    dataset_dir = os.path.join(VQA_DATASET_DIR, "datasets", dataset_name)
    question_file = DATASET_CONFIG[dataset_name][split]["question_file"]
    annotation_file = DATASET_CONFIG[dataset_name][split]["annotation_file"]
    image_dir = DATASET_CONFIG[dataset_name][split]["image_root"]
    coco_prefix = DATASET_CONFIG[dataset_name][split]["image_prefix"]

    question_data_fpath = os.path.join(dataset_dir, question_file)
    annotation_data_fpath = os.path.join(dataset_dir, annotation_file)
    image_root = os.path.join(COCO_DATASET_DIR, "images", image_dir)

    chunked_question_ids = get_chunked_question_ids(dataset_dir, split, chunk_id)

    with open(question_data_fpath) as f:
        question_data = json.load(f)

    with open(annotation_data_fpath) as f:
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


def load_gqa_dataset_from_json(dataset_name: str, split: str) -> List[Dict]:
    dataset_dir = os.path.join(VQA_DATASET_DIR, "datasets")
    annotation_file = DATASET_CONFIG[dataset_name][split]["annotation_file"]
    image_dir = DATASET_CONFIG[dataset_name][split]["image_root"]

    annotation_data_fpath = os.path.join(dataset_dir, dataset_name, annotation_file)
    image_root = os.path.join(dataset_dir, dataset_name, image_dir)

    with open(annotation_data_fpath) as f:
        annotated_data = json.load(f)

    dataset = []
    for idx in annotated_data:
        qa_obj = annotated_data[idx]
        imageId = qa_obj["imageId"]
        question = qa_obj["question"]
        answer = qa_obj["answer"]
        image_path = os.path.join(image_root, imageId + ".jpg")
        data = {
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "question_id": idx,
        }
        dataset.append(data)

    logger.info(json.dumps(dataset[0], indent=4))

    return dataset


def load_visual7w_dataset_from_json(dataset_name: str, split: str, multiple_choice: bool = True) -> List[Dict]:
    dataset_root = "datasets"
    fname = os.path.join(VQA_DATASET_DIR, dataset_root, dataset_name, "dataset.json")
    dataset = json.load(open(fname, "r"))
    dataset_split = defaultdict(list)
    for img in dataset["images"]:
        dataset_split[img["split"]].append(img)

    dataset_dir = os.path.join(VQA_DATASET_DIR, "datasets")
    image_dir = DATASET_CONFIG[dataset_name]["image_root"]
    img_prefix = DATASET_CONFIG[dataset_name]["image_prefix"]
    image_root = os.path.join(dataset_dir, dataset_name, image_dir)

    dataset = []
    for img in tqdm(dataset_split[split], desc=f"Preprocessing {dataset_name} dataset"):
        for pair in img["qa_pairs"]:
            qa_obj = {}
            image_id = pair["image_id"]
            image_path = os.path.join(image_root, img_prefix + str(image_id) + ".jpg")
            qa_obj["image_path"] = image_path
            qa_obj["question"] = pair["question"]
            choices = pair["multiple_choices"] + [pair["answer"]]
            qa_obj["choice"] = choices
            qa_obj["question_id"] = pair["qa_id"]
            qa_obj["answer"] = pair["answer"]
            dataset.append(qa_obj)

    if split == "train":
        random.shuffle(dataset)
        dataset = dataset[:5000]
        logger.warning("WARNING: Truncating train dataset to 5000 samples")

    logger.info(json.dumps(dataset[0], indent=4))

    return dataset
