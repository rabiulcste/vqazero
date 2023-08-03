import os
import json
from prompts.templates import DatasetTemplates


class PromptingHandler:
    def __init__(self, dataset_name: str, prompt_name=None, subset_name=None):
        self.prompt = None
        self.prompt_name = prompt_name
        self.gpt3_response_dir = os.path.join("output", dataset_name, "gpt3-api", "generated-questions")
        self.prompt = self.load_prompt_template(dataset_name, prompt_name, subset_name=subset_name)

    def generic_prompting_handler_for_winoground(self, example):
        if self.prompt is None:
            raise ValueError("Prompt template not loaded.")

        # apply the prompt to the example
        examples = [{"caption": example["caption_0"]}, {"caption": example["caption_1"]}]
        questions = []
        for ex in examples:
            questions.append(self.prompt.apply(ex)[0])  # apply the prompt to one example
        return questions

    def generic_prompting_handler_for_vqa(self, example):
        if self.prompt is None:
            raise ValueError("Prompt template not loaded.")

        return self.prompt.apply(example)[0]

    def decomposition_prompting_handler(self, example):
        if self.prompt is None:
            raise ValueError("Prompt template not loaded.")

        examples = [{"statement": example[f"caption_{i}"]} for i in range(2)]
        questions = []
        for ex in examples:
            questions.append(self.prompt.apply(ex)[0])  # apply the prompt to one example
        return questions

    def gpt3_baseline_qa_prompting_handler_for_winoground(self, example):
        cached_questions = self.load_cached_questions_for_winoground(example["id"])
        examples = [{"question": cached_questions[f"caption_{i}"]} for i in range(2)]
        questions = []
        for ex in examples:
            questions.append(self.prompt.apply(ex)[0])
        return questions

    def load_prompt_template(self, dataset_name, prompt_name, subset_name=None):
        prompts = DatasetTemplates(dataset_name=dataset_name, subset_name=subset_name)
        # print all the prompt names
        print(f"Available template for {dataset_name} dataset, names: {prompts.all_template_names}")
        # Select a prompt by its name
        if not prompt_name:
            prompt_name = "caption_question_answer_does_caption_match"
        prompt = prompts[prompt_name]
        return prompt

    def load_cached_questions_for_winoground(self, example_id):
        cached_question_dir = "prefix_convert_question_to_binary_vqa"
        fname = os.path.join(self.gpt3_response_dir, cached_question_dir, f"{example_id}.json")
        with open(fname, "r") as f:
            return json.load(f)


class DecompositionHandler:
    """
    A class for loading and handling the decomposition output from the GPT-3 API.
    Attributes:
        args (object): The object containing command-line arguments.
        prompt_name (str): The name of the prompt to be loaded.
        gpt3_response_dir (str): The directory containing the GPT-3 API output.
        data (dict): The decomposition output loaded from the JSON file.

    Output:
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
    "caption_1": { .... }
    }
    """

    def __init__(self, dataset_name: str, prompt_name: str, gen_model_name: str):
        self.dataset_name = dataset_name
        self.prompt_name = prompt_name
        self.gen_model_name = gen_model_name
        self.gpt3_response_dir = os.path.join(
            "output", self.dataset_name, "gpt3-api", "generated-subquestions"
        )
        self.data = self.load_subquestions_output_json()

    def get_gpt3_converted_question(self, example, handler: PromptingHandler):
        questions = handler.gpt3_baseline_qa_prompting_handler_for_winoground(example)
        return questions

    def get_cot_prompt(
        self, example_id: int, caption_id: int, decomposed_questions, handler: PromptingHandler
    ):
        cot_prompt = "You are given an image. Answer the following yes/no question about the image:\n"
        # now we need the question.
        # we can get the question from the prompt
        example = {"id": example_id}
        questions = self.get_gpt3_converted_question(example, handler)

        cot_prompt += f"{questions[caption_id]}\n"  # only this question is different in the whole prompt
        to_answer_txt = "To answer the above question, we will first answer the following sub-questions one-by-one in the image:"
        cot_prompt += to_answer_txt + "\n"
        for subquestion in decomposed_questions:
            cot_prompt += f"{subquestion}\n"
        cot_prompt += "Let's think step-by-step in the image.\n"
        return cot_prompt

    def load_subquestions_output_json(self):
        fname = os.path.join(self.gpt3_response_dir, self.prompt_name, self.gen_model_name, "output.json")
        with open(fname, "r") as f:
            output_json = json.load(f)
        return output_json

    def get_subquestions_data(self, id):
        return self.data[str(id)]
