from typing import List, Union

import torch
from PIL import Image


# flamingo preparation code
class FlamingoProcessor:
    def __init__(self, tokenizer, image_processor, device, cast_dtype):
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.cast_dtype = cast_dtype

    def _prepare_images(self, batch: List[List[Image.Image]]) -> torch.Tensor:
        """
        Convert images to tensors, reshape them, and stack them.
        Args:
            batch: A list of lists of images.
        Returns:
            preprocessed images (tensors) or None
                shape (B, T_img, F, C, H, W)
                None if no images in batch
        """
        images_per_example = max(len(x) for x in batch)
        batch_images = None
        for iexample, example in enumerate(batch):
            for iimage, image in enumerate(example):
                preprocessed = self.image_processor(image)
                if batch_images is None:
                    batch_images = torch.zeros(
                        (len(batch), images_per_example, 1) + preprocessed.shape,
                        dtype=preprocessed.dtype,
                    )
                batch_images[iexample, iimage, 0] = preprocessed

        return batch_images

    def _prepare_text(
        self,
        batch: List[str],
        padding="longest",
        truncation=True,
        max_length=2000,
    ):
        """
        Tokenize the text and stack them.
        Args:
            batch: A list of lists of strings.
        Returns:
            input_ids (tensor)
                shape (B, T_txt)
            attention_mask (tensor)
                shape (B, T_txt)
        """
        encodings = self.tokenizer(
            batch,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
            max_length=max_length,
        )
        input_ids, attention_mask = encodings["input_ids"], encodings["attention_mask"]

        return input_ids, attention_mask.bool()


from LLaVA.llava.constants import (DEFAULT_IM_END_TOKEN,
                                   DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN,
                                   IMAGE_TOKEN_INDEX)
from LLaVA.llava.conversation import conv_templates
from LLaVA.llava.mm_utils import tokenizer_image_token


class LlaVaProcessor:
    def __init__(self, tokenizer, image_processor, mm_use_im_start_end):
        self.mm_use_im_start_end = mm_use_im_start_end
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.conv_mode = "llava_v1"

    def load_demo_images(image_files: Union[List[str], str]):
        if type(image_files) is list:
            out = []
            for image_file in image_files:
                image = Image.open(image_file).convert("RGB")
                out.append(image)
        else:
            out = Image.open(image_files).convert("RGB")
        return out

    # TODO: refactor this, not working
    def get_processed_tokens_demo(self, text: str, image_files: Union[List[str], str]):
        if self.mm_use_im_start_end:
            qs = (
                qs
                + "\n"
                + DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                + DEFAULT_IM_END_TOKEN
                + "\n"
                + DEFAULT_IM_START_TOKEN
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                + DEFAULT_IM_END_TOKEN
            )
        else:
            qs = (
                qs
                + "\n"
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
                + "\n"
                + DEFAULT_IMAGE_PATCH_TOKEN * image_token_len
            )

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        images = self.load_demo_images(image_files)
        image_tensor = torch.stack(
            [self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0] for image in images]
        )

        input_ids = (
            tokenizer_image_token(text, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        )

        return image_tensor, input_ids

    def format_text(self, text: str):
        if self.mm_use_im_start_end:
            text = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + text
        else:
            text = DEFAULT_IMAGE_TOKEN + "\n" + text

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], text)
        conv.append_message(conv.roles[1], None)
        text = conv.get_prompt()

        return text

    def load_image(self, image_path: str):
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def pad_sequence_to_max_length(sequence, max_length, padding_value=0):
        """Pad a sequence to the desired max length."""
        if len(sequence) >= max_length:
            return sequence
        return torch.cat([sequence, torch.full((max_length - len(sequence),), padding_value, dtype=sequence.dtype)])

    def get_processed_tokens(self, text: str, image_path: str):
        prompt = self.format_text(text)
        image = self.load_image(image_path)

        input_ids = tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0)
        image_tensor = self.image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]

        return image_tensor, input_ids

    def get_processed_tokens_batch(self, batch_text: List[str], image_paths: List[str]):
        prompt = [self.format_text(text) for text in batch_text]
        images = [self.load_image(image_path) for image_path in image_paths]

        batch_input_ids = [
            tokenizer_image_token(prompt, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt") for prompt in prompt
        ]

        # Determine the maximum length of input_ids in the batch
        max_len = max([len(seq) for seq in batch_input_ids])
        # Pad each sequence in input_ids to the max_len
        padded_input_ids = [self.pad_sequence_to_max_length(seq.squeeze(), max_len) for seq in batch_input_ids]
        batch_input_ids = torch.stack(padded_input_ids)

        batch_image_tensor = self.image_processor(images, return_tensors="pt")["pixel_values"]

        return batch_image_tensor, batch_input_ids
