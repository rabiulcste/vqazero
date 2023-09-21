import json
import os
import random
from datetime import date, datetime

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evals.answer_postprocess import (clean_last_word,
                                      postprocess_batch_vqa_generation_blip2)
from evals.answer_postprocess_vicuna_llm import AnswerPostProcessLLM
from utils.logger import Logger

random.seed(42)

logger = Logger(__name__)


class CustomJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        # Handle numpy arrays
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        # Handle numpy numbers
        if isinstance(
            obj,
            (
                np.int_,
                np.intc,
                np.intp,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64,
            ),
        ):
            return int(obj)

        if isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        # Handle pandas DataFrame and Series
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="records")

        if isinstance(obj, pd.Series):
            return obj.to_dict()

        # Handle datetime objects
        if isinstance(obj, (datetime, date)):
            return obj.isoformat()

        # Handle other non-serializable objects or custom classes
        # By default, convert them to string (change this if needed)
        if hasattr(obj, "__dict__"):
            return obj.__dict__

        return super().default(obj)


class PredictionsDataset(Dataset):
    def __init__(self, predictions):
        self.qids = list(predictions.keys())
        self.data = [predictions[qid] for qid in self.qids]

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        data = self.data[idx]
        qid = self.qids[idx]
        input_text = (
            data["question"]
            + " "
            + data["raw_prediction"].capitalize()
            + ("." if not data["raw_prediction"].endswith(".") else "")
        )

        # Ensure input texts are not too long
        if len(input_text) > 1200:
            input_text = input_text[:1200]

        return {"qid": qid, "data": data, "input_text": input_text}


def collate_fn(batch):
    return {
        "qids": [item["qid"] for item in batch],
        "data": [item["data"] for item in batch],
        "input_text": [item["input_text"] for item in batch],
    }


def extract_answers_from_predictions_vicunallm(
    args, predictions, answer_extractor: AnswerPostProcessLLM, batch_size=32, num_examples_in_task_prompt=8, chunk_id=-1
):
    dataset = PredictionsDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    parsed_predictions = {}

    for batch in tqdm(dataloader, desc="Post-processing answers (vicuna)"):
        all_prompts = [
            answer_extractor._prepare_prompt(input_text, num_examples_in_task_prompt)
            for input_text in batch["input_text"]
        ]
        input_ids = answer_extractor.tokenizer(
            all_prompts, return_tensors="pt", padding=True, truncation=True
        ).input_ids.to(answer_extractor.device)
        generated_outputs = answer_extractor.generate_output_batch(
            input_ids=input_ids,
            max_length=50,
            num_return_sequences=1,
        )

        # Create a list to store the final results
        batch_answers = []
        for i, output in enumerate(generated_outputs):
            text = output[len(all_prompts[i]) :]
            split_text = text.split(
                f"{num_examples_in_task_prompt+2}. {answer_extractor.prompt_data['io_structure']['input_keys']}:"
            )
            if len(split_text) > 1:
                desired_text = split_text[0]
            else:
                desired_text = text
            batch_answers.append(desired_text.replace("\n", "").strip())

        batch_answers = postprocess_batch_vqa_generation_blip2(args.dataset_name, batch_answers)
        batch_answers = [clean_last_word(ans) for ans in batch_answers]

        for qid, curr_data, input_text, ans in zip(batch["qids"], batch["data"], batch["input_text"], batch_answers):
            curr_data.update({"prediction": ans, "input_for_vicuna": input_text})
            parsed_predictions[qid] = curr_data
            logger.debug(f"question_id = {qid}, input_text = {input_text}, prediction = {ans}")

    return parsed_predictions
