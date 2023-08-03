import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from modeling_utils import get_most_common_item
from utils.config import SCRATCH_DIR
from utils.globals import DATASET_CONFIG

THRESHOLD_MAP = {
    "gqa": 0.6,
    "aokvqa": 0.6,
    "okvqa": 0.6,
    "visual7w": 0.6,
    "vqa_v2": 0.6,
}


class NearestNeighborQuestionFinder:
    def __init__(self, data: Dict[str, List[str]], threshold: float = 0.7, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name, device="cpu")
        self.question_list = data["question"]
        self.answer_list = data["answer"]
        self.choice_list = data["choice"]
        self.context_list = data["context"]
        self.threshold = threshold
        self.question_embeddings = self.model.encode(self.question_list, convert_to_tensor=True)

    def find_nearest_neighbors(self, question: str, k: int = 5) -> List[Dict[str, str]]:
        question_embedding = self.model.encode(question, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(question_embedding, self.question_embeddings, dim=1)
        nearest_neighbors_indices = similarities.argsort(descending=True)

        nearest_neighbors = []
        unique_questions = {question}
        for i in nearest_neighbors_indices:
            if (
                similarities[i] < self.threshold
                and len(nearest_neighbors) < k
                and self.question_list[i] not in unique_questions
            ):
                support_example = {"question": self.question_list[i], "answer": self.answer_list[i]}
                if self.choice_list:
                    support_example["choice"] = self.choice_list[i]
                if self.context_list:
                    support_example["context"] = self.context_list[i]
                nearest_neighbors.append(support_example)
                unique_questions.add(self.question_list[i])
            elif len(nearest_neighbors) == k:
                break
        return nearest_neighbors

    def find_nearest_neighbors_batch(self, questions: List[str], k: int = 10) -> List[List[str]]:
        question_embeddings = self.model.encode(questions, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(question_embeddings, self.question_embeddings, dim=1)

        nearest_neighbors_indices = similarities.argsort(descending=True)[:, :k]
        nearest_neighbors = [[self.questions_list[i] for i in indices] for indices in nearest_neighbors_indices]
        return nearest_neighbors


split_map = {
    "train": "train",
    "val": "",
    "testdev": "testdev",
}


def load_caption(config, split, prompt_name="prefix_promptcap"):
    fpath = os.path.join(
        SCRATCH_DIR,
        "output",
        "generated_caption_dumps",
        config.dataset_name,
        config.gen_model_name,
        config.vqa_format,
        prompt_name,
        split_map[split],
        "output.json",
    )
    caption_data = json.load(open(fpath))
    print(f"Nearest neighbour search > Loaded {len(caption_data)} examples from {fpath}")
    return caption_data


def load_rationale(config, split, prompt_name):
    print(config.dataset_name, config.model_name, prompt_name, split)
    fpath = os.path.join(
        SCRATCH_DIR,
        "output",
        "generated_rationale_dumps",
        config.dataset_name,
        config.model_name,
        config.vqa_format,
        prompt_name,
        split_map[split],
        "output.json",
    )
    rationale_data = json.load(open(fpath))

    return rationale_data


def load_gqa_dataset_from_json(config, split, prompt_name) -> List[Dict]:
    dataset_name = config.dataset_name
    vqa_format = config.vqa_format
    dataset_dir = os.path.join(SCRATCH_DIR, "datasets")
    annotation_file = DATASET_CONFIG[dataset_name][split]["annotation_file"]
    annotation_data_fpath = os.path.join(dataset_dir, dataset_name, annotation_file)

    with open(annotation_data_fpath) as f:
        annotated_data = json.load(f)

    context_data = None
    if vqa_format == "caption_qa":
        context_data = load_caption(config, split, prompt_name)
    elif vqa_format == "cot_qa":
        context_data = load_rationale(config, split, prompt_name)

    dataset = []
    for idx in annotated_data:
        qa_obj = annotated_data[idx]
        qa_obj["questionId"] = idx
        if context_data is not None:
            if str(idx) not in context_data:
                continue
            qa_obj["context"] = context_data[str(idx)]
        dataset.append(qa_obj)
    return dataset


def load_vqa_dataset_from_json(config, split, prompt_name) -> List[Dict]:
    dataset_name = config.dataset_name
    vqa_format = config.vqa_format

    dataset_dir = os.path.join(SCRATCH_DIR, "datasets")
    question_file = DATASET_CONFIG[dataset_name][split]["question_file"]
    annotation_file = DATASET_CONFIG[dataset_name][split]["annotation_file"]

    question_data_fpath = os.path.join(dataset_dir, dataset_name, question_file)
    annotation_data_fpath = os.path.join(dataset_dir, dataset_name, annotation_file)

    with open(question_data_fpath) as f:
        question_data = json.load(f)

    with open(annotation_data_fpath) as f:
        answer_data = json.load(f)
        answer_data = answer_data["annotations"]

    answer_data_dict = {}
    for answer_obj in answer_data:
        question_id = answer_obj["question_id"]
        answer_data_dict[question_id] = answer_obj

    # TODO: support rationale too here
    # we need to know whether it is CoT or Caption-qa
    context_data = None
    if vqa_format == "caption_qa":
        context_data = load_caption(config, split, prompt_name)
    elif vqa_format == "cot_qa":
        context_data = load_rationale(config, split, prompt_name)

    dataset = []
    for qa_obj in tqdm(question_data["questions"], desc=f"Preprocessing {dataset_name} dataset"):
        question_id = qa_obj["question_id"]
        # adding answer to the question object
        answer_data = answer_data_dict[question_id]
        if context_data is not None:
            if str(question_id) not in context_data:
                continue
            qa_obj["context"] = context_data[str(question_id)]
        qa_obj.update(answer_data)
        dataset.append(qa_obj)

    if split == "train":
        random.shuffle(dataset)
        dataset = dataset[:10000]
    print(json.dumps(dataset[0], indent=4))
    return dataset


def load_visual7w_dataset_from_json(config, split_name, prompt_name):
    dataset_name = config.dataset_name
    vqa_format = config.vqa_format
    dataset_root = "datasets"
    fname = os.path.join(SCRATCH_DIR, dataset_root, dataset_name, "dataset.json")
    dataset = json.load(open(fname, "r"))
    split = defaultdict(list)
    for img in dataset["images"]:
        split[img["split"]].append(img)

    context_data = None
    if vqa_format == "caption_qa":
        context_data = load_caption(config, split_name, prompt_name)
    elif vqa_format == "cot_qa":
        context_data = load_rationale(config, split_name, prompt_name)

    dataset = []
    for img in tqdm(split[split_name], desc=f"Preprocessing {dataset_name} dataset"):
        for pair in img["qa_pairs"]:
            qa_obj = {}
            qa_obj["question"] = pair["question"]
            choices = pair["multiple_choices"] + [pair["answer"]]
            random.shuffle(choices)
            qa_obj["choice"] = choices
            qa_obj["question_id"] = pair["qa_id"]
            qa_obj["answer"] = pair["answer"]

            if context_data is not None:
                if str(qa_obj["question_id"]) not in context_data:
                    continue
                qa_obj["context"] = context_data[str(qa_obj["question_id"])]
            dataset.append(qa_obj)

    print(json.dumps(dataset[0], indent=4))
    return dataset


def get_aokvqa_dataset(config, split, prompt_name, version="v1p0"):
    dataset_name = config.dataset_name
    vqa_format = config.vqa_format
    assert split in ["train", "val", "test", "test_w_ans"]
    dataset_dir = os.path.join(SCRATCH_DIR, "datasets", dataset_name)
    dataset = json.load(open(os.path.join(dataset_dir, f"aokvqa_{version}_{split}.json")))

    context_data = None
    if vqa_format == "caption_qa":
        context_data = load_caption(config, split, prompt_name)
    elif vqa_format == "cot_qa":
        context_data = load_rationale(config, split, prompt_name)

    qa_objects = []
    for qa_obj in tqdm(dataset, desc=f"Preprocessing {dataset_name} dataset"):
        question_id = qa_obj["question_id"]
        question = qa_obj["question"]
        choices = qa_obj["choices"]
        if config.task_type == "multiple_choice":
            answer = choices[qa_obj["correct_choice_idx"]]  # hacky way to get the answer
        else:
            answer = get_most_common_item(qa_obj["direct_answers"])
        random.shuffle(choices)

        qa_dict = {
            "question_id": question_id,
            "question": question,
            "answer": answer,
        }
        if config.task_type == "multiple_choice":
            qa_dict["choice"] = choices

        if context_data is not None:
            if str(question_id) not in context_data:
                continue
            qa_dict["context"] = context_data[str(question_id)]

        qa_objects.append(qa_dict)

    return qa_objects


class KNNHandler:
    def __init__(self, config):
        self.config = config
        split = "train"
        if "," in config.prompt_name:
            prompt_name = config.prompt_name.split(",")[1]
        else:
            prompt_name = config.prompt_name
        self.dataset_name = config.dataset_name
        self.vqa_format = config.vqa_format
        dataset = self.get_dataset(config, split, prompt_name)
        self.knn_qa = self.setup_knn(dataset)

    def get_dataset(self, config, split, prompt_name):
        dataset_name = config.dataset_name
        if dataset_name == "aokvqa":
            dataset = get_aokvqa_dataset(config, split, prompt_name)
        elif dataset_name in ["okvqa", "vqa_v2"]:
            dataset = load_vqa_dataset_from_json(config, split, prompt_name)
        elif dataset_name == "gqa":
            dataset = load_gqa_dataset_from_json(config, split, prompt_name)
        elif dataset_name == "visual7w":
            dataset = load_visual7w_dataset_from_json(config, split, prompt_name)
        else:
            raise NotImplementedError(f"Dataset {config.dataset_name} not implemented")
        return dataset

    def setup_knn(self, dataset):
        question_list = []
        answer_list = []
        choice_list = []
        context_list = []
        for idx in range(len(dataset)):
            content = dataset[idx]
            if self.dataset_name == "carets":
                question = content["sent"]
                answer = list(content["label"].keys())
            elif self.dataset_name in ["vqa_v2", "okvqa"]:
                answer = content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
                question = content["question"]
            else:
                question = content["question"]
                answer = content["answer"]

            choices = content.get("choice", None)
            if choices:
                last_choice = choices.pop()
                options = f"{', '.join(choices)} or {last_choice}?"
                choice_list.append(options)

            context = content.get("context", None)
            if context:
                context_list.append(context)

            question_list.append(question)
            answer_list.append(answer)
        data_dict = {
            "question": question_list,
            "answer": answer_list,
            "choice": choice_list,
            "context": context_list,
        }
        threshold = THRESHOLD_MAP[self.dataset_name]
        return NearestNeighborQuestionFinder(data=data_dict, threshold=threshold)

    def format_support_example(self, support_ex, vqa_format):
        if vqa_format in ["caption_qa", "basic_qa"]:
            context = support_ex.get("context", "")
            context_str = f"{context} " if context else ""
            if "short_answer" in self.config.prompt_name:
                return f"Context: {context_str} Question: {support_ex['question']} {support_ex.get('choice', '')} Short answer: {support_ex['answer']} "
            else:
                return f" {context_str} {support_ex['question']} {support_ex.get('choice', '')} {support_ex['answer']}"
        elif vqa_format == "cot_qa":
            return f"Q: {support_ex.get('question', '')} A: {support_ex['context']}"
        else:
            raise ValueError(f"Invalid vqa_format: {vqa_format}")

    def find_nearest_neighbors(self, question):
        nearest_neighbors = self.knn_qa.find_nearest_neighbors(question)
        support_questions = [self.format_support_example(ex, self.vqa_format) for ex in nearest_neighbors]
        random.shuffle(support_questions)
        support_questions = "\n".join(support_questions)
        instructions = "In this task, your goal is to write an answer to a question about the image.\n---\nTo write the answer, here are some sample QA suggestions (not related to the given image):\n"
        # instructions = "In this task, your goal is to write an answer to a question about the image. Here are some suggested QA pairs (not related to the given image)\n"
        support_questions = instructions + support_questions + "\nNow answer the following question about the image."
        return support_questions
