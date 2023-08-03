import glob
import json
import os
import pprint
import random
import re
import pandas as pd
from typing import Dict, Tuple

import openai
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    OPTForCausalLM,
    T5ForConditionalGeneration,
)
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff


from utils import config
from utils.globals import MODEL_CLS_INFO
from utils.logger import Logger
from utils.handler import PromptingHandler

# Initialize the GPT3API config
api_key = config.OPENAI_API_KEY
organization = config.ORGANIZATION

openai.api_key = api_key
openai.organization = organization

logger = Logger(__name__)

# TODO: Integrate LLM with 8bit optimizer https://github.com/huggingface/transformers/issues/20361
MODEL_CLASSES = {
    "gptj": (AutoModelForCausalLM, AutoTokenizer, "EleutherAI/gpt-j-6B"),
    "t5": (T5ForConditionalGeneration, AutoTokenizer, "t5-large"),
    "flant5": (AutoModelForSeq2SeqLM, AutoTokenizer, "google/flan-t5-xxl"),
    "opt": (OPTForCausalLM, AutoTokenizer, "facebook/opt-13b"),
}


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


@retry(wait=wait_random_exponential(min=1, max=5), stop=stop_after_attempt(3))
def chat_completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


completion_with_backoff(model="text-davinci-003", prompt="Once upon a time,")


class QuestionGeneratorforWinoground:
    def __init__(self, model_type):
        self.model_type = model_type
        self.load_model_and_processor()

    def load_model_and_processor(self):
        if self.model_type in ["gpt3", "chatgpt"]:
            logger.info("WARNING: Money is burning!!!")
            self.model_name = self.model_type
            self.tokenizer = None
        elif self.model_type in ["gptj", "t5", "flant5", "opt"]:
            model_class, tokenizer_class, model_name_or_path = MODEL_CLASSES[self.model_type]
            self.model = model_class.from_pretrained(model_name_or_path)
            self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
            self.model_name = model_name_or_path
        else:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. Must be one of {MODEL_CLASSES.keys()}."
            )

    def generate_questions(self, winoground_dataset: Dataset, handler: PromptingHandler, output_dir: str):
        output_json = {}
        for example_id, example in enumerate(tqdm(winoground_dataset)):
            save_path = os.path.join(output_dir, f"{example['id']}.json")
            if os.path.exists(save_path):
                logger.info(f"Already generated questions for id {example['id']}")
                continue

            output_data = {}
            logger.info(f"Generating questions for id {example['id']}")
            try:
                prompt_engineered_captions = handler.generic_prompting_handler(example)
                logger.info(f"Modified captions: {pprint.pformat(prompt_engineered_captions)}")
                prompt_data = {
                    "caption_0": prompt_engineered_captions[0],
                    "caption_1": prompt_engineered_captions[1],
                }
                for key in prompt_data:
                    prompt = prompt_data[key]
                    if self.model_type == "gpt3":
                        response = completion_with_backoff(
                            model="text-davinci-003",
                            prompt=prompt,
                            temperature=0.7,
                            max_tokens=256,
                        )
                        generated_question = response["choices"][0]["text"].strip()
                    elif self.model_type == "chatgpt":
                        # Note: you need to be using OpenAI Python v0.27.0 for the code below to work
                        response = chat_completion_with_backoff(
                            model="gpt-3.5-turbo",
                            messages=[
                                # {"role": "system", "content": "You are a helpful assistant."},
                                {"role": "user", "content": prompt},
                            ],
                        )
                        generated_question = response["choices"][0]["message"]["content"].strip()
                    elif self.model_type == "gptj" or "opt":
                        input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
                        generated_tokens = self.model.generate(
                            input_ids=input_ids,
                            max_length=256,
                            temperature=0.7,
                            do_sample=True,
                            num_return_sequences=1,
                        )
                        generated_question = self.tokenizer.decode(
                            generated_tokens[0],
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False,
                        ).strip()

                    output_data[key] = {
                        "caption": example[key],
                        "prompt": prompt,
                        "question": generated_question,
                    }
                output_json[example["id"]] = output_data
                with open(save_path, "w") as f:
                    json.dump(output_json, f, indent=2)

            except Exception as e:
                logger.info(f"Error generating questions for id {example['id']}: {e}")
            break

    def get_generated_questions_for_winoground(self, example_id, output_dir):
        file_path = os.path.join(output_dir, f"{example_id}.json")
        if os.path.exists(file_path):
            with open(file_path) as f:
                output_json = json.load(f)
                return output_json
        else:
            logger.info(f"No generated questions found for id {example_id}")
            return None

    def stepwise_generate_decomposed_prompting_output_for_winoground(self, example):
        # step 1: generate a literal description of the image
        literal_desc = self.generate_literal_desc(example)

        # step 2: generate a set of questions without the image
        questions = self.generate_decompose_questions(literal_desc)

        # step 3: build a rational statement based on the generated response
        rational_statement = self.build_rational_statement(questions)

        return questions, rational_statement

    def generate_literal_desc(self, example):
        prompt_literal_desc = self.prompt.templates[0]
        desc = prompt_literal_desc.apply(example)[0]

        response = self.gpt.submit_request(prompt=prompt_literal_desc.prompt + desc, max_tokens=100)
        literal_desc = response.choices[0].text.strip()

        return literal_desc

    def generate_decompose_questions(self, literal_desc):
        prompt_questions = self.prompt.templates[1]
        prompt = prompt_questions.prompt.replace("LITERAL_DESC", literal_desc)

        response = self.gpt.submit_request(prompt=prompt, max_tokens=100)

        questions = []
        for choice in response.choices:
            question = choice.text.strip()
            questions.append(question)

        return questions

    def build_rational_statement(self, prompt, questions):
        prompt_rational = self.prompt.templates[2]
        prompt = prompt_rational.prompt.replace("QUESTION_0", questions[0]).replace(
            "QUESTION_1", questions[1]
        )
        response = self.gpt.submit_request(prompt=prompt, max_tokens=100)
        rational_statement = response.choices[0].text.strip()
        return rational_statement


class DecomposeGeneratorForWinoground:

    """
    This class is used to generate decomposed subquestions for winoground dataset.

    The below data format is used to store the generated questions and rational statement for each example in winoground dataset.
    {
        "caption_0": {
        "caption": "an old person kisses a young person",
        "prompt1": "'an old person kisses a young person' What it means in terms of an image literally?",
        "respons1": "In terms of an image literally, it would be an image of an elderly person kissing a young person on the cheek or forehead.",
        "prompt2": "'an old person kisses a young person' What it means in terms of an image literally? \n\nIn terms of an image literally, it would be an image of an elderly person kissing a young person on the cheek or forehead. \n\nWrite some questions to query on the image. This question and corresponding answers will be used to verify if the caption matches the image. Finally, create a step-by-step rationale statement based on the answer to verify the caption matches the image.",
        "decomposed_questions": [
            "Is the image of an elderly person?",
            "Is the image of a young person?",
            "Are the two people in the image kissing?"
        ],
        "decomposed_answers": [
            "Yes",
            "Yes",
            "Yes"
        ],
        "rationales": "Rationale Statement: Based on the answers to the questions, the image is of an elderly person and a young person kissing, which matches the caption 'an old person kisses a young person', thus verifying"
        },
        "caption_1": {...},
        "decomposed_vqa_answers": {
        "c0_i0": [
            "yes",
            "no",
            "yes"
        ],
        "c0_i1": [...],
        "c1_i0": [...],
        "c1_i1": [...]
        }
    }
    """

    def __init__(self, dataset_name, gen_model_name: str, model_name: str = None):
        self.dataset_name = dataset_name
        self.gen_model_name = gen_model_name  # the generator model name e.g. chatgpt
        self.model_name = model_name  # the vqa  model name e.g. blip2
        self.vqa_format = "decompose_qa"
        self.question_data_dir = os.path.join(
            "output", dataset_name, "gpt3-api", "generated-subquestions"
        )  # "generated_subquestions" is the folder name to store the generated subquestions
        self.vqa_data_dir = os.path.join(
            "output",
            self.dataset_name,
            self.model_name,
            "decompose_qa",
        )
        self.data = None
        self.load_model_and_processor()

    def load_model_and_processor(self):
        if self.gen_model_name in ["gpt3", "chatgpt"]:
            logger.info("WARNING: Money is burning!!!")
            self.tokenizer = None
        elif self.gen_model_name in ["gptj", "t5", "flant5", "opt"]:
            model_class, tokenizer_class, model_name_or_path = MODEL_CLASSES[self.model_type]
            self.model = model_class.from_pretrained(model_name_or_path)
            self.tokenizer = tokenizer_class.from_pretrained(model_name_or_path)
        else:
            raise ValueError(
                f"Invalid model_type '{self.model_type}'. Must be one of {MODEL_CLASSES.keys()}."
            )

    def print_sample_generated_questions(self, prompt_name: str):
        subquestion_dir = os.path.join(self.question_data_dir, prompt_name, self.gen_model_name, "by_id")
        subquestion_files = os.listdir(subquestion_dir)

        # select a random file from the list of files
        subquestion_file = random.choice(subquestion_files)
        subquestion_file_path = os.path.join(subquestion_dir, subquestion_file)

        with open(subquestion_file_path, "r") as f:
            subquestions_data = json.load(f)

        example_id = subquestions_data["id"]
        caption_0_question = subquestions_data["caption_0"]["decomposed_questions"]
        caption_1_question = subquestions_data["caption_1"]["decomposed_questions"]
        logger.info("=====================================================")
        logger.info("=====================================================")
        logger.info(f"Example id: {example_id}")
        logger.info(f"Generated sub-questions for example {example_id}:")
        logger.info(f"Caption 0 sub-questions: {pprint.pformat(caption_0_question)}")
        logger.info(f"Caption 1 sub-questions: {pprint.pformat(caption_1_question)}")
        logger.info("=====================================================")

    # method to generate decomposed subquestions for a given example
    def generate_decomposed_subquestions(self, winoground_dataset: Dataset, handler: PromptingHandler):
        for example_id, example in enumerate(tqdm(winoground_dataset)):
            # if example_id > 50: # for testing purposes we only run on 50 examples
            #     break

            save_path = os.path.join(
                self.question_data_dir,
                handler.prompt_name,
                self.gen_model_name,
                "by_id",
                f"{example['id']}.json",
            )  # save to "by_id" subdir

            if os.path.exists(save_path):
                logger.info(f"Already generated questions for id {example['id']} at {save_path}")
                continue

            output_data = {"id": example["id"]}
            logger.info(f"Generating questions for id {example['id']}")

            try:
                prompt_engineered_captions = handler.decomposition_prompting_handler(example)

            except Exception as e:
                logger.error(f"Error generating prompt for id {example['id']}: {e}")
                continue

            logger.info(f"Modified captions: {pprint.pformat(prompt_engineered_captions)}")

            prompt_data = {
                "caption_0": prompt_engineered_captions[0],
                "caption_1": prompt_engineered_captions[1],
            }

            for key, prompt in prompt_data.items():
                try:
                    generated_questions = self._generate_questions(prompt)
                except Exception as e:
                    logger.error(f"Error generating questions for id {example['id']}: {e}")
                    continue

                decomposed_question_list = generated_questions.split("\n")

                output_data[key] = {
                    "caption": example[key],
                    "prompt": prompt,
                    "decomposed_questions": decomposed_question_list,
                }
            self._save_output_json(output_data, save_path)
        self._aggregate_by_id_json_files_to_output_json(self.question_data_dir, handler.prompt_name)

    def _generate_questions(self, prompt: str) -> str:
        if self.gen_model_name == "gpt3":
            response = completion_with_backoff(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=256,
            )
            generated_questions = response["choices"][0]["text"].strip()

        elif self.gen_model_name == "chatgpt":
            response = chat_completion_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            generated_questions = response["choices"][0]["message"]["content"].strip()

        elif self.gen_model_name in ["gptj", "opt"]:
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt")
            generated_tokens = self.model.generate(
                input_ids=input_ids,
                max_length=256,
                temperature=0.7,
                do_sample=True,
                num_return_sequences=1,
            )
            generated_questions = self.tokenizer.decode(
                generated_tokens[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            ).strip()

        else:
            raise ValueError(f"Invalid model_type: {self.gen_model_name}")

        return generated_questions

    # load the generated decomposed subquestions from the output file
    def _get_input_data_for_rationale_generation(self, prompt_name: str, decomposition_type: str):
        fname = os.path.join(self.vqa_data_dir, prompt_name, decomposition_type, "output.json")
        with open(fname, "r") as f:
            output_json = json.load(f)
        return output_json

    # get the generated decomposed subquestions for a given example id
    def get_input_data_for_rationale_generation_by_id(self, prompt_name, decomposition_type, id):
        if self.data is None:
            self.data = self._get_input_data_for_rationale_generation(prompt_name, decomposition_type)
        return self.data[str(id)]

    # load the generated decomposed subquestions from the output file
    def _get_vqa_prediction_data(self, prompt_name: str, decomposition_type: str):
        fname = os.path.join(self.vqa_data_dir, prompt_name, decomposition_type, self.gen_model_name, "output_final.json")
        with open(fname, "r") as f:
            output_json = json.load(f)
        return output_json

    # call the API to generate rationales on the vqa outputs: decomposed subquestions and corresponding answers
    def generate_rationale(self, prompt_name: str, decomposition_type: str):
        json_path = os.path.join(self.vqa_data_dir, prompt_name, decomposition_type, self.gen_model_name, "output_final.json")

        if os.path.exists(json_path):
            logger.info(f"Rationale already exists at {json_path}. Skipping...")
            return

        final_instruction_txt = (
            "\n Now generate the rationale and final answer based on the above instruction, subquestions and "
            "corresponding answers. Let's think step-by-step. Answer yes if stamenet matches the image, otherwise no. Output in this format \n Rationale: \n Final Answer: (yes/no) "
        )
        WINOGROUND_TEST_SIZE = 400

        for wid in range(WINOGROUND_TEST_SIZE):
            data = self.get_input_data_for_rationale_generation_by_id(
                prompt_name, decomposition_type, str(wid)
            )
            logger.debug(json.dumps(data, indent=4))

            save_path = os.path.join(
                self.vqa_data_dir,
                prompt_name,
                decomposition_type,
                self.gen_model_name,
                "by_id",
                f"{data['id']}.json",
            )  # save to "by_id" subdir

            if os.path.exists(save_path):
                logger.info(f"Already generated questions for id {data['id']} at {save_path}")
                continue

            curr_rationale, curr_final_answer, curr_yes_no_answer = {}, {}, {}
            pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

            for cid, iid in pairs:
                prompt = self._create_prompt_for_rationale_generation(data, cid, iid, final_instruction_txt)

                if self.gen_model_name in ["chatgpt", "gpt3"]:
                    text = self._generate_rationale_and_final_answer(prompt)
                else:
                    raise NotImplementedError("Only ['chatgpt', 'gpt3'] is supported for now.")

                logger.info(f"COMPLETION TEXT: {text}")

                (
                    rationale,
                    final_answer,
                    yes_no_answer,
                ) = self._parse_rationale_and_final_answer(text)

                curr_rationale[f"c{cid}_i{iid}"] = rationale
                curr_final_answer[f"c{cid}_i{iid}"] = final_answer
                curr_yes_no_answer[f"c{cid}_i{iid}"] = yes_no_answer

            self._update_output_data(
                data,
                curr_rationale,
                curr_final_answer,
                curr_yes_no_answer,
                final_instruction_txt,
            )
            self._save_output_json(data, save_path)

        self._aggregate_by_id_json_files_to_output_json(
            self.vqa_data_dir, prompt_name, suffix="_final", decomposition_type=decomposition_type
        )
        logger.info(f"Rationale generation for {prompt_name} is done.")

    def _create_prompt_for_rationale_generation(
        self, data: Dict, cid: int, iid: int, final_instruction_txt: str
    ) -> str:
        task_instruction_txt = "Your task is to verify a user's statement regarding an image. To do this, you will be provided with a series of subquestions and answers from a VQA model about the image. Your goal is to use these pre-collected answers to match the user statement with the image. Pay attention to the word order in the user statement and ignore any questions that are not relevant to matching the user statement. To complete the task, provide a final response along with reasoning based on the given subquestions and answers."
        # Your task is to verify a user's statement regarding an image, using a set of subquestions and their respective answers to verify whether the statement matches the image. Note that VQA models are not perfect, so they may not be able to answer all questions 100% correct. However, if you have strong evidence from a subset of the questions and the information is sufficient, you may use them in reaching a final decision. To accomplish this, you need to provide a rationale and a final answer that are based on the subquestions and their answers provided below."

        user_statement = data[f"caption_{cid}"]["caption"]
        curr_decomposed_questions = data[f"caption_{cid}"]["decomposed_questions"]
        curr_decomposed_answers = data["decomposed_vqa_answers"][f"c{cid}_i{iid}"]

        logger.debug(f"curr_decomposed_questions: {curr_decomposed_questions}")
        logger.debug(f"curr_decomposed_answers: {curr_decomposed_answers}")

        qa_pairs = [
            f"{q} \n {a} \n "
            for qaid, (q, a) in enumerate(zip(curr_decomposed_questions, curr_decomposed_answers))
        ]
        qa_pairs = "".join(qa_pairs)
        logger.debug(f"qa_pairs: {qa_pairs}")
        prompt = (
            task_instruction_txt
            + "\n User statement:  "
            + user_statement
            + "\n Subquestions and answers: \n"
            + qa_pairs
            + final_instruction_txt
        )
        logger.debug(f"prompt: {prompt}")
        return prompt

    def _generate_rationale_and_final_answer(self, prompt: str) -> str:
        if self.gen_model_name == "gpt3":
            response = completion_with_backoff(
                model="text-davinci-003",
                prompt=prompt,
                temperature=0.7,
                max_tokens=512,
            )
            text = response["choices"][0]["text"].strip()
        else:
            response = chat_completion_with_backoff(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )
            text = response["choices"][0]["message"]["content"].strip()

        return text

    def _aggregate_by_id_json_files_to_output_json(
        self, data_dir, prompt_name: str, suffix: str = "", decomposition_type: str = "",
    ):
        """
        the output.json file will be in the format of
        {
          "0": {
              "caption_0": {
                "caption": "an old person kisses a young person",
                "prompt": "You are required to use a VQA system to verify if a user's statement matches a given image word-for-word. To accomplish this task, the user's complex statement will be simplified into a series of sub-questions of increasing complexity, allowing for a step-by-step reasoning process. Try to imagine such an image. The aim is to achieve an exact word-for-word match between the statement and the image. The sub-questions will be utilized to obtain a response from the VQA model, which will provide a final verdict.\n\nUser statement: \"A young person kisses an old person.\"\nSub-questions:\nQ1: find the young person\nQ2: find the old person\nQ3: Where is the kiss taking place, on the lips or the cheek?\nQ4: Is it the young person who is kissing the old person?\nQ5: Who is initiating the kiss?\n\nUser statement: \"an old person kisses a young person\"\nSub-questions:",
                "decomposed_questions": [
                    "Q1: find the young person",
                    "Q2: find the old person",
                    "Q3: Where is the kiss taking place, on the lips or the cheek?",
                    "Q4: Is it the old person who is kissing the young person?",
                    "Q5: Who is initiating the kiss?"
                ],
            },
        }
        """
        all_json_paths = glob.glob(
            os.path.join(data_dir, prompt_name, decomposition_type, self.gen_model_name, "by_id", "*.json")
        )
        output_json = {}
        for json_path in all_json_paths:
            with open(json_path, "r") as f:
                data = json.load(f)
            output_json[data["id"]] = data
        output_json_path = os.path.join(data_dir, prompt_name, decomposition_type, self.gen_model_name, f"output{suffix}.json")

        logger.info(f"Saved aggregated json at {output_json_path}")
        self._save_output_json(output_json, output_json_path)

    def generate_eval(self, prompt_name: str, decomposition_type: str):
        json_path = os.path.join(self.vqa_data_dir, prompt_name, decomposition_type, self.gen_model_name, "output_final.json")
        if not os.path.exists(json_path):
            logger.info(f"Final answer file doesn't exist at {json_path}. Skipping...")
            return

        WINOGROUND_TEST_SIZE = 400

        winoground_scores = []
        data = self._get_vqa_prediction_data(prompt_name, decomposition_type)
        for wid in range(WINOGROUND_TEST_SIZE):
            try:
                winoground_scores.append(data[str(wid)]["yes_no_answer"])
            except KeyError:
                logger.info(f"KeyError for wid {wid}")
                continue

        self._compute_eval(winoground_scores, prompt_name, decomposition_type)

    # TODO: duplicate code from lavis_blip/inference.py
    def _compute_eval(self, winoground_blip_scores, prompt_name: str, decomposition_type: str):
        # Convert winoground_blip_scores to pandas DataFrame and save it as a CSV file
        df = pd.DataFrame(winoground_blip_scores)
        csv_path = os.path.join(
            self.vqa_data_dir, prompt_name, decomposition_type, "winoground_blip_score.csv"
        )
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved the VQA output to {csv_path}")

        # Define three functions to evaluate the results
        def group_correct(result):
            return (
                result["c0_i0"] == "yes"
                and result["c1_i0"] == "no"
                and result["c1_i1"] == "yes"
                and result["c0_i1"] == "no"
            )

        def common_correct(result):
            return result["c0_i0"] == "yes" and result["c0_i1"] == "no"

        def uncommon_correct(result):
            return result["c1_i1"] == "yes" and result["c1_i0"] == "no"

        group_correct_count = 0
        common_correct_count = 0
        uncommon_correct_count = 0
        for result in winoground_blip_scores:
            _result = {}
            for key, sentence in result.items():
                if key == "id":
                    continue
                match = re.search(r"^\w+", str(sentence))  # find the first word that matches the pattern
                if match:
                    _result[key] = match.group().lower()
                else:
                    _result[key] = sentence

            group_correct_count += 1 if group_correct(_result) else 0
            common_correct_count += 1 if common_correct(_result) else 0
            uncommon_correct_count += 1 if uncommon_correct(_result) else 0

        denominator = len(winoground_blip_scores)
        logger.info(f"group score: {group_correct_count * 100 / denominator}")
        logger.info(f"common score: {common_correct_count * 100 / denominator}")
        logger.info(f"uncommon score: {uncommon_correct_count * 100 / denominator}")

        # Save the results as a JSON file
        data = {
            "dataset_name": "winoground",
            "prompt_name": prompt_name,
            "template_expression": decomposition_type,
            "model_name": self.model_name,
            "vqa_format": self.vqa_format,
            "num_examples": len(winoground_blip_scores),
            "group_score": group_correct_count * 100 / denominator,
            "common_score": common_correct_count * 100 / denominator,
            "uncommon_score": uncommon_correct_count * 100 / denominator,
        }

        json_path = os.path.join(self.vqa_data_dir, prompt_name, decomposition_type, "result_meta.json")
        self._save_output_json(data, json_path)
        logger.info(f"Saved result information to {json_path}")

    @staticmethod
    def _parse_rationale_and_final_answer(text: str) -> Tuple[str, str, str]:
        rationale_start = text.find("Rationale:")
        rationale_end = text.find("Final Answer:")
        rationale = text[rationale_start:rationale_end].strip()

        final_answer_start = text.find("Final Answer:")
        final_answer = text[final_answer_start:].strip().lower()

        if "yes" in final_answer:
            yes_no_answer = "yes"
        elif "no" in final_answer:
            yes_no_answer = "no"
        else:
            yes_no_answer = "Unknown"

        return rationale, final_answer, yes_no_answer

    @staticmethod
    def _update_output_data(
        data: Dict,
        curr_rationale: Dict,
        curr_final_answer: Dict,
        curr_yes_no_answer: Dict,
        final_instruction_txt: str,
    ):
        data["rationale"] = curr_rationale
        data["answer"] = curr_final_answer
        data["yes_no_answer"] = curr_yes_no_answer
        data["final_instruction"] = final_instruction_txt

    @staticmethod
    def _save_output_json(output_json: Dict, json_path: str):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
        with open(json_path, "w") as f:
            json.dump(output_json, f, indent=4)

        logger.info(f"Saved the output JSON to {json_path}")


class CaptionGeneratorforWinoground:
    """
    A class for generating captions for the Winoground dataset.
    Only support LAVIS model for now.
    """

    def __init__(self, model_name, device):
        self.model_name = model_name
        self.output_dir = os.path.join("output", "generated_captions", self.model_name)
        self.dataset = load_dataset("facebook/winoground", use_auth_token=True)[
            "test"
        ]  # winoground-dataset
        self.model, self.vis_processors = self.load_model_and_processors(model_name, device)

    def load_model_and_processors(self, model_cls_name, device):
        """
        Load the model and processors for the given model class name.
        """
        MODEL_INFO = MODEL_CLS_INFO["lavis"]
        from lavis.models import load_model_and_preprocess

        if model_cls_name not in MODEL_INFO:
            raise ValueError(f"Invalid `args.model_name`. Provided: {model_cls_name}")

        logger.info(f"loading model and processor for `{model_cls_name}`")
        model_info = MODEL_INFO[model_cls_name]
        model_name = model_info["name"]
        model_type = model_info.get("model_type")

        model, vis_processors, _ = load_model_and_preprocess(
            name=model_name,
            model_type=model_type,
            is_eval=True,
            device=device,
        )
        return model, vis_processors

    def get_generated_captions(self, example_id):
        file_path = os.path.join(self.output_dir, f"{example_id}.json")
        if os.path.exists(file_path):
            with open(file_path) as f:
                output_json = json.load(f)
                return output_json
        else:
            logger.info(f"No generated questions found for id {example_id}")
            return None

    def generate_captions(self):
        """
        Generate captions for the Winoground dataset using the BLIP2 model.
        Saves the generated captions for each example to a JSON file.
        """
        for eid, example in enumerate(tqdm(self.dataset)):
            logger.info(f"Generating captions for id {example['id']}")
            NUM_IMAGES = 2
            images = [example[f"image_{i}"].convert("RGB") for i in range(NUM_IMAGES)]
            image_tensors = [self.vis_processors["eval"](img) for img in images]

            samples = {
                "image": torch.stack(image_tensors).to(self.model.device),
                "prompt": "",
            }

            if self.model_name == "blip_caption":
                samples = {"image": torch.stack(image_tensors).to(self.model.device)}

            model_generated_captions = self.model.generate(
                samples=samples,
                repetition_penalty=1.5,
            )

            if eid < 3:
                logger.info(f"Generated captions for id {eid}")
                logger.info(json.dumps(model_generated_captions, indent=4))

            output_json = {
                "id": example["id"],
                "caption_0": model_generated_captions[0],
                "caption_1": model_generated_captions[1],
            }

            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Save the generated captions to a JSON file
            save_path = os.path.join(self.output_dir, f"{example['id']}.json")
            with open(save_path, "w") as f:
                json.dump(output_json, f, indent=2)

            logger.info(f"Saved generated captions for id {example['id']} to {save_path}")
