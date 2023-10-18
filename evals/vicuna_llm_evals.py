import random
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from vllm import SamplingParams

from evals.answer_postprocess import (clean_last_word, majority_vote,
                                      postprocess_vqa_answers)
from evals.vicuna_llm import LlmEngine
from utils.logger import Logger

random.seed(42)

logger = Logger(__name__)


class PredictionsDataset(Dataset):
    def __init__(self, predictions):
        self.qids = list(predictions.keys())
        self.data = [predictions[qid] for qid in self.qids]

    def __len__(self):
        return len(self.qids)

    def __getitem__(self, idx):
        data = self.data[idx]
        qid = self.qids[idx]
        input_text = self._get_input_text(data)

        return {"qid": qid, "data": data, "input_text": input_text}

    # TODO: we can add `multiple_choice` logic here if needed
    def _get_input_text(self, data):
        is_self_consistency = "reasoning_paths" in data
        if is_self_consistency:
            reasoning_paths = data["reasoning_paths"]
            return [self._format_input_text(data["question"], path) for path in reasoning_paths]

        return [self._format_input_text(data["question"], data["raw_prediction"])]

    @staticmethod
    def _format_input_text(question, additional_text):
        # Ensure input texts are not too long
        max_len = 1200
        formatted_text = f"{question} {additional_text.capitalize()}"
        if not additional_text.endswith("."):
            formatted_text += "."

        return formatted_text[:max_len]


def collate_fn(batch):
    return {
        "qids": [item["qid"] for item in batch],
        "data": [item["data"] for item in batch],
        "input_text": [item["input_text"] for item in batch],
    }


def extract_answers_from_predictions_vicunallm_legacy(
    predictions, llm_engine: LlmEngine, batch_size=16, num_examples_in_task_prompt=8, chunk_id=-1
):
    dataset = PredictionsDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    parsed_predictions = {}

    for batch in tqdm(dataloader, desc="Post-processing answers (vicuna)"):
        all_prompts = [
            llm_engine._prepare_prompt(input_text, num_examples_in_task_prompt) for input_text in batch["input_text"]
        ]
        input_ids = llm_engine.tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True).input_ids.to(
            llm_engine.device
        )

        generated_outputs = llm_engine.generate_output_batch(
            input_ids=input_ids,
            max_length=10,
            num_return_sequences=1,
        )

        # Create a list to store the final results
        extracted_answers = [
            extract_desired_text(ans, num_examples_in_task_prompt, llm_engine).replace("\n", "").strip()
            for ans in generated_outputs
        ]

        for qid, curr_data, input_text, ans in zip(
            batch["qids"], batch["data"], batch["input_text"], extracted_answers
        ):
            curr_data.update({"prediction": ans, "input_for_vicuna": input_text})
            parsed_predictions[qid] = curr_data
            logger.debug(f"question_id = {qid}, input_text = {input_text}, prediction = {ans}")

    return parsed_predictions


def generate_output_for_input_texts(
    input_texts, llm_engine: LlmEngine, max_gpu_batch, max_length=10, num_examples_in_task_prompt=8
):
    all_generated_outputs = []
    for i in range(0, len(input_texts), max_gpu_batch):
        chunked_input_text = input_texts[i : i + max_gpu_batch]
        all_prompts = [llm_engine._prepare_prompt(text, num_examples_in_task_prompt) for text in chunked_input_text]
        if llm_engine.vllm_enabled:
            sampling_params = SamplingParams(max_tokens=max_length)
            generated_outputs = llm_engine.llm.generate(
                prompts=all_prompts, sampling_params=sampling_params, use_tqdm=False
            )
            generated_outputs = [output.outputs[0].text for output in generated_outputs]

        else:
            device = llm_engine.llm.device
            encodings = llm_engine.tokenizer(all_prompts, return_tensors="pt", padding=True, truncation=True)
            input_ids = encodings.input_ids.to(device)
            attention_mask = encodings.attention_mask.to(device)

            generated_outputs = llm_engine.generate_output_batch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_length,
            )
        all_generated_outputs.extend(generated_outputs)

    assert len(all_generated_outputs) == len(input_texts)

    return all_generated_outputs


def extract_desired_text(ans_text, num_examples_in_task_prompt, llm_engine):
    split_text = ans_text.split(
        f"{num_examples_in_task_prompt+2}. {llm_engine.prompt_data['io_structure']['input_keys']}:"
    )
    return split_text[0] if len(split_text) > 1 else ans_text


def get_single_or_first_element(lst):
    """
    Return the first element if the list has more than one element.
    If the list has only one element, return that element.
    In all other cases, return the input as-is.
    """
    if not isinstance(lst, list):
        return lst
    if not lst:  # empty list
        return lst
    return lst[0] if len(lst) == 1 else lst


def extract_answers_from_predictions_vicuna(
    predictions,
    llm_engine: LlmEngine,
    batch_size=16,
    num_examples_in_task_prompt=8,
    chunk_id: int = -1,
):
    dataset = PredictionsDataset(predictions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)

    parsed_predictions = {}

    for batch in tqdm(dataloader, desc="Post-processing answers (vicuna)"):
        is_2d_input = isinstance(batch["input_text"][0], list)

        if is_2d_input:
            flat_input_text = [text for sublist in batch["input_text"] for text in sublist]
        else:
            flat_input_text = batch["input_text"]

        all_generated_outputs = generate_output_for_input_texts(
            flat_input_text, llm_engine, batch_size, num_examples_in_task_prompt=num_examples_in_task_prompt
        )
        processed_answers = [
            extract_desired_text(ans, num_examples_in_task_prompt, llm_engine).replace("\n", "").strip()
            for ans in all_generated_outputs
        ]

        if is_2d_input:
            reshaped_outputs = [
                processed_answers[sum(map(len, batch["input_text"][:i])) : sum(map(len, batch["input_text"][: i + 1]))]
                for i in range(len(batch["input_text"]))
            ]
        else:
            reshaped_outputs = processed_answers

        for qid, curr_data, input_texts, answers in zip(
            batch["qids"], batch["data"], batch["input_text"], reshaped_outputs
        ):
            # answers might be a list of strings (1d) or a list of lists of strings (2d)
            prediction = majority_vote(answers) if is_2d_input else answers
            prediction = get_single_or_first_element(prediction)
            input_texts = get_single_or_first_element(input_texts)
            curr_data.update({"vicuna_prediction": answers, "input_for_vicuna": input_texts, "prediction": prediction})
            parsed_predictions[qid] = curr_data
            logger.debug(f"question_id = {qid}, input_text = {input_texts}, prediction = {prediction}")

    return parsed_predictions


def postprocess_cleanup_vicuna(dataset_name, parsed_predictions):
    # post-process the predictions
    batch_answers = [data["prediction"] for data in parsed_predictions.values()]
    batch_questions = [data["question"] for data in parsed_predictions.values()]
    batch_answers = postprocess_vqa_answers(dataset_name, batch_answers, batch_questions)
    batch_answers = [clean_last_word(ans) for ans in batch_answers]

    # update parsed_predictions with cleaned answers
    for qid, cleaned_ans in zip(parsed_predictions.keys(), batch_answers):
        parsed_predictions[qid]["prediction"] = cleaned_ans

    return parsed_predictions
