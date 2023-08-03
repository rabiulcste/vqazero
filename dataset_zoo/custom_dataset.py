import json
import os
import random
import unittest
from collections import defaultdict
from string import punctuation
from typing import Dict, List

import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from carets.carets.dataset import CaretsDataset
from modeling_utils import get_most_common_item
from nearest_neighbor import KNNHandler
from utils.config import SCRATCH_DIR
from utils.globals import DATASET_CONFIG

random.seed(42)


class VQADataset(Dataset):
    def __init__(self, config, dataset_name: str, split: str = "val", processor=None, prompt_handler=None, **kwargs):
        self.config = config
        self.split = split
        self.dataset_name = dataset_name
        self.dataset = self.get_dataset()
        self.processor = processor
        self.prompt_handler = prompt_handler
        self.model_name = kwargs.get("model_name")
        self.caption_gen_cls = kwargs.get("caption_gen_cls")
        self.cot_gen_cls = kwargs.get("cot_gen_cls")

        if self.caption_gen_cls:
            print(f"Provided caption_gen_cls: {self.caption_gen_cls}\n" f"Loading cached caption from disk...")
            self.caption_gen_cls.load(self.split)

        elif self.cot_gen_cls:
            print(f"Provided cot_gen_cls: {self.cot_gen_cls}")
            print("Loading cached cot from disk...")
            self.cot_gen_cls.load(self.split)

        self.knn_handler = None
        if self.prompt_handler and "knn" in self.prompt_handler.prompt_name:
            self.knn_handler = KNNHandler(config)

        elif self.cot_gen_cls:
            print(f"Provided cot_gen_cls: {self.cot_gen_cls}")
            print("Loading cached cot from disk...")
            self.cot_gen_cls.load()

        self.knn_handler = None
        if self.prompt_handler and "knn" in self.prompt_handler.prompt_name:
            self.knn_handler = KNNHandler(config)

    def get_dataset(self):
        if self.dataset_name == "aokvqa":
            dataset = self.load_aokvqa_dataset_from_json()
        elif self.dataset_name == "visual7w":
            dataset = self.load_visual7w_dataset_from_json()
        elif self.dataset_name in ["vqa_v2", "okvqa"]:
            dataset = self.load_vqa_dataset_from_json()
        elif self.dataset_name == "carets":
            dataset = CaretsDataset("./carets/configs/default.yml").splits[self.split].questions
        elif self.dataset_name == "gqa":
            dataset = self.load_gqa_dataset_from_json()
        else:
            raise NotImplementedError(f"Dataset {self.dataset_name} is not implemented yet.")

        return dataset

    def load_vqa_dataset_from_json(self) -> List[Dict]:
        dataset_dir = os.path.join(SCRATCH_DIR, "datasets")
        question_file = DATASET_CONFIG[self.dataset_name][self.split]["question_file"]
        annotation_file = DATASET_CONFIG[self.dataset_name][self.split]["annotation_file"]
        image_dir = DATASET_CONFIG[self.dataset_name][self.split]["image_root"]
        coco_prefix = DATASET_CONFIG[self.dataset_name][self.split]["image_prefix"]

        question_data_fpath = os.path.join(dataset_dir, self.dataset_name, question_file)
        annotation_data_fpath = os.path.join(dataset_dir, self.dataset_name, annotation_file)
        image_root = os.path.join(dataset_dir, "coco_images", image_dir)

        # vqa v2 chunked dataset
        if self.config.chunk_id is not None:
            chunked_question_ids_fpath = os.path.join(dataset_dir, self.dataset_name, "chunked_question_ids.json")
            with open(chunked_question_ids_fpath) as f:
                chunked_data = json.load(f)
            chunked_question_ids = chunked_data[self.config.chunk_id]

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
        for qa_obj in tqdm(question_data["questions"], desc=f"Preprocessing {self.dataset_name} dataset"):
            image_id = qa_obj["image_id"]
            question_id = qa_obj["question_id"]
            if self.dataset_name == "vqa_v2" and self.config.chunk_id is not None and self.split != "train":
                if question_id not in chunked_question_ids:
                    continue
            image_path = os.path.join(image_root, coco_prefix + "{:012d}".format(image_id) + ".jpg")
            qa_obj.update({"image_path": image_path})  # adding image path to the question object

            # adding answer to the question object
            answer_data = answer_data_dict[question_id]
            qa_obj.update(answer_data)
            dataset.append(qa_obj)
        if self.split == "train":
            random.shuffle(dataset)
            dataset = dataset[:10000]
        print(json.dumps(dataset[0], indent=4))
        return dataset

    def load_gqa_dataset_from_json(self) -> List[Dict]:
        dataset_dir = os.path.join(SCRATCH_DIR, "datasets")
        annotation_file = DATASET_CONFIG[self.dataset_name][self.split]["annotation_file"]
        image_dir = DATASET_CONFIG[self.dataset_name][self.split]["image_root"]

        annotation_data_fpath = os.path.join(dataset_dir, self.dataset_name, annotation_file)
        image_root = os.path.join(dataset_dir, self.dataset_name, image_dir)

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
        print(json.dumps(dataset[0], indent=4))
        return dataset

    def load_aokvqa_dataset_from_json(self, version="v1p0") -> List[Dict]:
        """
        Builds a dataset from a JSON file using the AOKVQA dataset format.
        Returns:
            A list of dictionaries, where each dictionary represents a QA object.
            Each QA object has the following keys: 'question_id', 'question', 'image_path', and 'direct_answers'.
        """
        assert self.split in ["train", "val", "test", "test_w_ans"]
        dataset_dir = os.path.join(SCRATCH_DIR, "datasets", self.dataset_name)
        dataset = json.load(open(os.path.join(dataset_dir, f"aokvqa_{version}_{self.split}.json")))
        coco_dir = os.path.join(SCRATCH_DIR, "datasets", "coco_images")

        qa_objects = []
        for qa_obj in tqdm(dataset, desc=f"Preprocessing {self.dataset_name} dataset"):
            image_id = qa_obj["image_id"]
            question_id = qa_obj["question_id"]
            question = qa_obj["question"]
            choices = qa_obj["choices"]
            answer = get_most_common_item(qa_obj["direct_answers"])  # hacky way to get the answer
            random.shuffle(choices)
            image_path = self._get_coco_path(self.split, image_id, coco_dir)
            qa_dict = {
                "question_id": question_id,
                "question": question,
                "image_path": image_path,
                "answer": answer,
                "choice": choices,
            }
            qa_objects.append(qa_dict)

        print(json.dumps(qa_objects[0], indent=4))
        return qa_objects

    def load_visual7w_dataset_from_json(self):
        dataset_root = "datasets"
        fname = os.path.join(SCRATCH_DIR, dataset_root, self.dataset_name, "dataset.json")
        dataset = json.load(open(fname, "r"))
        split = defaultdict(list)
        for img in dataset["images"]:
            split[img["split"]].append(img)

        dataset_dir = os.path.join(SCRATCH_DIR, "datasets")
        image_dir = DATASET_CONFIG[self.dataset_name]["image_root"]
        img_prefix = DATASET_CONFIG[self.dataset_name]["image_prefix"]
        image_root = os.path.join(dataset_dir, self.dataset_name, image_dir)

        dataset = []
        for img in tqdm(split[self.split], desc=f"Preprocessing {self.dataset_name} dataset"):
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

        if self.split == "train" and len(dataset) > 12000:
            random.shuffle(dataset)
            dataset = dataset[:12000]
            print("WARNING: Truncating train dataset to 12000 samples")

        print(json.dumps(dataset[0], indent=4))
        return dataset

    def get_image_qa_multiple_choice_dataset(self, shuffle=False):
        dataset_root = "datasets"
        fname = os.path.join(SCRATCH_DIR, dataset_root, self.dataset_name, "dataset.json")
        dataset = json.load(open(fname, "r"))
        split = defaultdict(list)
        for img in dataset["images"]:
            split[img["split"]].append(img)

        dataset = {}
        for i, img in enumerate(split[self.split]):
            for pair in img["qa_pairs"]:
                qa_obj = {}
                qa_obj["image_id"] = pair["image_id"]
                qa_obj["question"] = pair["question"]
                qa_obj["question_id"] = pair["qa_id"]
                qa_obj["answer"] = pair["answer"]
                qa_obj["mc"] = [pair["answer"]] + pair["multiple_choices"]
                if shuffle:
                    random.shuffle(qa_obj["mc"])
                qa_obj["mc_selection"] = qa_obj["mc"].index(pair["answer"])
                dataset[qa_obj["question_id"]] = qa_obj
        return dataset

    @staticmethod
    def _get_coco_path(split, image_id, coco_dir):
        return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")

    def __repr__(self) -> str:
        return f"{self.dataset_name} {self.split} dataset with {len(self)} examples"

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx) -> Dict:
        content = self.dataset[idx]
        question_id = content["question_id"]
        image_path = content["image_path"]
        if self.dataset_name == "carets":
            question = content["sent"]
            answer = list(content["label"].keys())
        elif self.dataset_name in ["vqa_v2", "okvqa"]:
            answer = content["answers"][0].get("raw_answer", content["answers"][0].get("answer"))
            question = content["question"]
        else:
            question = content["question"]
            answer = content["answer"]

        if self.dataset_name in ["visual7w", "aokvqa"]:
            choice = content["choice"]
            choice = [c.strip(punctuation) for c in choice]
            random.shuffle(choice)
            lst_choice = choice.pop()  # remove the last element and store it in a separate variable
            choice = ", ".join(choice) + " or " + lst_choice + "?"

        image = Image.open(image_path).convert("RGB")
        data = {
            "image": image,
            "image_path": image_path,
            "question": question,
            "answer": answer,
            "question_id": question_id,
        }

        if self.prompt_handler is not None:
            inp = {"question": question, "add_special_tokens": "flamingo" in self.model_name}
            if self.dataset_name in ["visual7w", "aokvqa"] and self.config.task_type == "multiple_choice":
                inp.update({"choice": choice})

            # TODO: this only works for standard-qa setting
            if self.knn_handler is not None:
                support_questions = self.knn_handler.find_nearest_neighbors(question)

            if self.caption_gen_cls:
                prefix_caption = self.caption_gen_cls.load_by_ids(str(question_id))
                inp.update({"caption": prefix_caption})
            elif self.cot_gen_cls:
                prefix_rationale = self.cot_gen_cls.load_by_ids(str(question_id))
                inp.update({"rationale": prefix_rationale})
            prompted_question = self.prompt_handler.generic_prompting_handler_for_vqa(inp)
            if "opt" in self.config.model_name:
                prompted_question = prompted_question.replace("\n", " ")

            if self.knn_handler is not None:
                prompted_question = f"{support_questions} {prompted_question}"
                # print(f"Prompted question: {prompted_question}")

            prompted_question = prompted_question.strip()
            data.update({"prompted_question": prompted_question})

        # this is a hacky way to add the processed image tensor and prompted question to the data
        if self.processor is not None:
            encoding = self.processor(images=image, padding="max_length", return_tensors="pt")
            encoding = {k: v.squeeze() for k, v in encoding.items()}
            data.update(**encoding)
        return data


def collate_fn(batch):
    # get the keys from the first batch element
    batch_keys = batch[0].keys()
    bkeys = ["question", "answer", "question_id", "prompted_question", "image", "image_path"]
    # Create the batch dictionary
    batch_dict = {}
    for bkey in bkeys:
        if bkey in batch_keys:
            batch_dict[bkey + "s"] = [example[bkey] for example in batch]

    return batch_dict


class TestVQA(unittest.TestCase):
    def setUp(self):
        self.dataset = VQADataset(dataset_name="okvqa")

    def test_length(self):
        self.assertEqual(
            len(self.dataset), 5046
        )  # replace 1000 with the actual number of examples in the validation set

    def test_example_keys(self):
        example = self.dataset[0]
        self.assertTrue("image" in example.keys())
        self.assertTrue("question" in example.keys())
        self.assertTrue("answer" in example.keys())
        self.assertTrue("question_id" in example.keys())
        self.assertTrue("id" in example.keys())

    def test_image(self):
        example = self.dataset[0]
        image = example["image"]
        self.assertEqual(image.size, (640, 480))  # replace (224, 224) with the actual size of the images in the dataset
        self.assertEqual(image.mode, "RGB")

    def test_collate(self):
        batch = [self.dataset[i] for i in range(5)]
        collated = collate_fn(batch)
        self.assertTrue("images" in collated.keys())
        self.assertTrue("questions" in collated.keys())
        self.assertTrue("answers" in collated.keys())
        self.assertEqual(len(collated["images"]), 5)
        self.assertEqual(len(collated["questions"]), 5)
        self.assertEqual(len(collated["answers"]), 5)
