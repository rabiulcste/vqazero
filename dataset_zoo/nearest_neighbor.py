import json
import os
from typing import Dict, List

import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from utils.logger import Logger
import random
from dataset_zoo.common import get_vqa_dataset
from utils.config import OUTPUT_DIR

THRESHOLD_MAP = {
    "gqa": 0.7,
    "aokvqa": 0.7,
    "okvqa": 0.7,
    "visual7w": 0.6,
    "vqa_v2": 0.6,
}

random.seed(42)

logger = Logger(__name__)


class NearestNeighborQuestionFinder:
    def __init__(self, data: Dict[str, List[str]], threshold: float = 0.7, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.sbert_model = SentenceTransformer(model_name, device="cuda")
        self.question_ids = data["question_ids"]
        self.questions = data["questions"]
        self.threshold = threshold
        logger.info(f"Computing embeddings for {len(self.questions)} questions...")
        self.question_embeddings = self.sbert_model.encode(self.questions, convert_to_tensor=True)

    def find_nearest_neighbors(self, question: str, k: int = 5) -> List[Dict[str, str]]:
        question_embedding = self.sbert_model.encode(question, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(question_embedding, self.question_embeddings, dim=1)
        nearest_neighbors_indices = similarities.argsort(descending=True)

        nearest_neighbor_ids = []
        unique_question_ids = set()
        for i in nearest_neighbors_indices:
            if (
                similarities[i] < self.threshold
                and len(nearest_neighbor_ids) < k
                and self.question_ids[i] not in unique_question_ids
            ):
                nearest_neighbor_ids.append(self.question_ids[i])
                unique_question_ids.add(self.question_ids[i])
            elif len(nearest_neighbor_ids) == k:
                break
        return nearest_neighbor_ids

    def find_nearest_neighbors_batch(self, questions: List[str], k: int = 10) -> List[List[str]]:
        question_embeddings = self.model.encode(questions, convert_to_tensor=True)
        similarities = torch.nn.functional.cosine_similarity(question_embeddings, self.question_embeddings, dim=1)

        nearest_neighbors_indices = similarities.argsort(descending=True)[:, :k]
        nearest_neighbors = [[self.questions_list[i] for i in indices] for indices in nearest_neighbors_indices]
        return nearest_neighbors


def cache_nearest_neighbor_data(dataset_name, multiple_choice=False):
    fpath = os.path.join(
        OUTPUT_DIR,
        "cache",
        "nearest_neighbors",
        dataset_name,
        "train_to_val",
        "output.json",
    )

    if os.path.exists(fpath):
        logger.info(f"Nearest neighbor data already exists at {fpath}")
        return

    question_list = []
    question_ids = []

    query_dataset = get_vqa_dataset(dataset_name, split="val", multiple_choice=multiple_choice)
    exemplar_dataset = get_vqa_dataset(dataset_name, split="train", multiple_choice=multiple_choice)
    for idx in range(len(exemplar_dataset)):
        content = exemplar_dataset[idx]

        question_id = content["question_id"]
        question = content["question"]

        question_ids.append(question_id)
        question_list.append(question)

    data_dict = {
        "question_ids": question_ids,
        "questions": question_list,
    }
    threshold = THRESHOLD_MAP[dataset_name]
    nn_search = NearestNeighborQuestionFinder(data=data_dict, threshold=threshold)

    query_support_dict = {}
    for query in tqdm(query_dataset, desc=f"Computing nearest neighbors for {dataset_name}..."):
        question = query["question"]
        question_id = query["question_id"]
        query_support_dict[str(question_id)] = nn_search.find_nearest_neighbors(question)

    os.makedirs(os.path.dirname(fpath), exist_ok=True)
    json.dump(query_support_dict, open(fpath, "w"))
    logger.info(f"Saved nearest neighbor data to {fpath}")
