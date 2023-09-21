import json
import os
import random
from typing import List

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)

from utils.logger import Logger

logger = Logger(__name__)

random.seed(42)


class AnswerPostProcessLLM:
    """
    A base class that uses a language model to generate output on a given input text from in-context examples.
    """

    def __init__(
        self,
        language_model_name: str,
        device: str,
    ):
        """
        Initializes the tokenizer and llm using the specified pretrained model.

        Args:
            language_model_name (str): The name of the pretrained model to use.
        """
        logger.info(
            f"Initializing {self.__class__.__name__} with language_model_name: {language_model_name}. It may take a few minutes to load the model."
        )

        self.language_model_name = language_model_name
        self.device = device

    def load_pretrained_model_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
        self.tokenizer.padding_side = "left"

        if "t5" in self.language_model_name:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.language_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.language_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto",
            )

    def _load_in_context_examples(self, dataset_name: str, output_mode: str):
        fname = os.path.join(
            "/home/mila/r/rabiul.awal/vqazero-private", "evals/demonstrations", "in_context_examples.json"
        )
        logger.info(f"Loading in-context examples from {fname}.")
        with open(fname, "r") as f:
            data = json.load(f)

        prompt_data = data[output_mode][dataset_name]
        self.prompt_data = prompt_data

    def _prepare_prompt(self, input_text: str, num_examples_in_task_prompt: int):
        selected_examples = self.prompt_data["samples"][:num_examples_in_task_prompt]
        random.shuffle(selected_examples)

        prompt_instruction = ""
        for i, example in enumerate(selected_examples, start=1):
            prompt_instruction += f"{i}. {example}\n"
        prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{input_text}} {{output_key}}:"
        prompt = prompt_instruction.format(
            input_key=self.prompt_data["io_structure"]["input_keys"],
            input_text=input_text,
            output_key=self.prompt_data["io_structure"]["output_keys"],
        )
        return prompt

    def generate_output(
        self,
        input_text: str,
        max_length: int = 50,
        num_examples_in_task_prompt: int = 8,
        num_return_sequences: int = 1,
    ):
        """
        Generates descriptive output given an input text and a list of examples.

        Args:
            input_text (str): The input text.
            examples (List[str]): The list of examples.
            max_length (int): The maximum length of the generated output.

        Returns:
            A list of generated outputs.
        """
        if not isinstance(input_text, str):
            raise ValueError("Input text must be a string.")

        prompt = self._prepare_prompt(input_text, num_examples_in_task_prompt)
        logger.debug(f"Input full prompt: {prompt}")

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
        prompt_length = input_ids.shape[-1]
        model_output = self.model.generate(
            input_ids,
            max_length=prompt_length + max_length,
            do_sample=True,
            top_k=40,
            temperature=0.8,
            num_return_sequences=num_return_sequences,
        )

        decoded_output = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

        generated_outputs = []
        for i, output in enumerate(decoded_output):
            text = output[len(prompt) :]
            split_text = text.split(
                f"{num_examples_in_task_prompt+2}. {self.prompt_data['io_structure']['input_keys']}:"
            )
            if len(split_text) > 1:
                desired_text = split_text[0]
            else:
                desired_text = text
            generated_outputs.append(desired_text.replace("\n", "").strip())

        logger.debug(f"Input Text: {input_text} | Generated Output: {generated_outputs}")
        return generated_outputs

    def generate_output_batch(
        self,
        input_ids: torch.Tensor,
        max_length=50,
        num_return_sequences: int = 1,
    ):
        """
        Generates descriptive tags for a batch of images given their captions.

        Args:
            input_ids (torch.Tensor): The input ids for the batch of texts.
            max_length (int): The maximum length of the generated tags.

        Returns:
            A list of lists, where each sub-list contains descriptive tags for the corresponding image.
        """

        prompt_length = input_ids.shape[-1]

        # Generate output for all prompts
        model_output = self.model.generate(
            input_ids,
            max_length=prompt_length + max_length,
            do_sample=True,
            top_k=40,
            temperature=0.8,
            num_return_sequences=num_return_sequences,
        )

        # Decode the output
        generated_outputs = self.tokenizer.batch_decode(model_output, skip_special_tokens=True)

        return generated_outputs
