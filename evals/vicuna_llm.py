import json
import os
import random
from typing import List, Union

import torch
from transformers import (AutoModelForCausalLM, AutoModelForSeq2SeqLM,
                          AutoTokenizer)
from vllm import LLM

from utils.config import PROJECT_DIR
from utils.logger import Logger

logger = Logger(__name__)

random.seed(42)

VLLM_MODELS = ["lmsys/vicuna-13b-v1.3", "meta-llama/Llama-2-70b-hf"]


class LlmEngine:
    """
    A base class that uses a language model to generate output on a given input text from in-context examples.
    """

    def __init__(
        self,
        language_model_name: str,
        device: str,
        fname: str = None,
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
        self.demonstrations_fname = self.get_demonstrations_fname(fname)
        self.device = device
        self.autocast_dtype = self._get_autocast_dtype()
        self.autocast_dtype_str = self._get_autocast_dtype_str()

    @staticmethod
    def get_demonstrations_fname(fname: [str, None]):
        return fname if fname else os.path.join(PROJECT_DIR, "evals/demonstrations", "in_context_examples.json")

    def _get_autocast_dtype(self):
        """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _get_autocast_dtype_str(self):
        """Return the best available dtype for autocasting: bfloat16 if available, else float16."""
        if torch.cuda.is_bf16_supported():
            return "bfloat16"
        return "float16"

    def load_pretrained_model_tokenizer(self):
        logger.info(f"Loading pretrained model and tokenizer: {self.language_model_name}")
        self.vllm_enabled = False

        if self.language_model_name in VLLM_MODELS:
            self.llm = LLM(
                model=self.language_model_name,
                dtype=self.autocast_dtype_str,
                tensor_parallel_size=torch.cuda.device_count(),
            )
            self.vllm_enabled = True

        else:
            self.tokenizer = AutoTokenizer.from_pretrained(self.language_model_name)
            self.tokenizer.padding_side = "left"
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({"pad_token": "</s>"})

            self.llm = AutoModelForCausalLM.from_pretrained(
                self.language_model_name,
                low_cpu_mem_usage=True,
                torch_dtype=self.autocast_dtype,
                trust_remote_code=True,
                device_map="auto",
            )
            self.vllm_enabled = False

    def _load_in_context_examples(self, **kwargs):
        logger.info(f"Loading in-context examples from {self.demonstrations_fname}.")
        with open(self.demonstrations_fname, "r") as f:
            data = json.load(f)

        dataset_name = kwargs.get("dataset_name", None)
        format_type = kwargs.get("format_type", "samples")
        if dataset_name:
            prompt_data = data[dataset_name]
        else:
            prompt_data = data

        self.prompt_data = prompt_data
        self.format_type = format_type

    def _prepare_prompt(self, input_text: str, num_examples_in_task_prompt: int, use_input_key: bool = True):
        selected_examples = self.prompt_data[self.format_type][:num_examples_in_task_prompt]
        random.shuffle(selected_examples)

        prompt_instruction = self.prompt_data.get("task_description", "")
        if prompt_instruction:
            prompt_instruction += "\n"

        for i, example in enumerate(selected_examples, start=1):
            if isinstance(example, dict):
                example = " ".join([f"{k}: {v}" for k, v in example.items()])
            prompt_instruction += f"{i}. {example}\n"

        if use_input_key:
            prompt_instruction += f"{len(selected_examples) + 1}. {{input_key}}: {{input_text}} {{output_key}}:"
        else:
            prompt_instruction += f"{len(selected_examples) + 1}. {{input_text}} {{output_key}}:"
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
        num_beams: int = 1,
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
        generated_ids = self.llm.generate(
            input_ids,
            max_length=prompt_length + max_length,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
        )
        generated_ids = generated_ids[:, prompt_length:]
        generated_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        logger.debug(f"Input Text: {input_text}, Generated Output: {generated_outputs}")
        return generated_outputs

    def generate_output_batch(
        self,
        input_ids: torch.Tensor,
        attention_mask: Union[torch.Tensor, None] = None,
        max_length: int = 50,
        num_beams: int = 1,
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
        generated_ids = self.llm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=prompt_length + max_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams,
        )
        generated_ids = generated_ids[:, prompt_length:]
        generated_outputs = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        return generated_outputs
