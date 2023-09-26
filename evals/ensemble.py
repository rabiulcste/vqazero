import argparse
import json
import os
from collections import defaultdict

from vqa_accuracy import eval_aokvqa, eval_gqa, eval_vqa

from utils.config import OUTPUT_DIR
from utils.globals import vqa_prompts


class EnsemblePredictions:
    def __init__(self, args):
        self.args = args
        self.identifier_key = "question" if self.args.vqa_format == "standard_vqa" else "caption"
        self.prompts = vqa_prompts
        self.data_dir_path = os.path.join(
            OUTPUT_DIR, "output", self.args.dataset_name, self.args.model_name, self.args.vqa_format
        )
        self.ensemble_predictions = defaultdict(lambda: {"prediction_collection": []})

    def gather_predictions(self):
        print(f"ALL PROMPTS = {json.dumps(self.prompts, indent=4)}")
        prompts_dirs = [os.path.join(self.data_dir_path, prompt) for prompt in self.prompts]

        for prompt_dir in prompts_dirs:
            fpath = os.path.join(prompt_dir, self.args.task_type, self.args.split_name, "predictions.json")

            if not os.path.exists(fpath):
                print(f"File not found: {fpath}")
                continue

            with open(fpath, "r") as f:
                data = json.load(f)

            for key in data:
                self.ensemble_predictions[key]["prediction_collection"].append(data[key]["prediction"])

    def add_gt_answer_to_predictions(self):
        prompts_dirs = [os.path.join(self.data_dir_path, prompt) for prompt in self.prompts]

        for prompt_dir in prompts_dirs:
            fpath = os.path.join(prompt_dir, self.args.task_type, self.args.split_name, "predictions.json")

            if os.path.exists(fpath):
                with open(fpath, "r") as f:
                    data = json.load(f)

                for key in data:
                    self.ensemble_predictions[key]["answer"] = data[key]["answer"]
                break

    def compute_ensemble(self):
        for key in self.ensemble_predictions:
            self.ensemble_predictions[key]["prediction"] = max(
                self.ensemble_predictions[key]["prediction_collection"],
                key=self.ensemble_predictions[key]["prediction_collection"].count,
            )

    def save_ensemble_predictions(self):
        ensemble_path = os.path.join(self.data_dir_path, "ensemble", self.args.task_type, "ensemble_predictions.json")
        output_dir = os.path.dirname(ensemble_path)
        os.makedirs(output_dir, exist_ok=True)

        with open(ensemble_path, "w") as f:
            json.dump(self.ensemble_predictions, f, indent=4)

        print(f"Saved ensemble predictions to {ensemble_path}")

    def run_evaluation(self):
        output_dir = os.path.dirname(
            os.path.join(self.data_dir_path, "ensemble", self.args.task_type, "ensemble_predictions.json")
        )
        if args.dataset_name == "aokvqa":
            eval_aokvqa(self.args, output_dir, self.ensemble_predictions)
        elif args.dataset_name == "okvqa":
            eval_vqa(self.args, output_dir)
        elif args.dataset_name == "gqa":
            self.add_gt_answer_to_predictions()
            eval_gqa(self.args, output_dir, results=self.ensemble_predictions)

    def process(self):
        self.gather_predictions()
        self.compute_ensemble()
        self.save_ensemble_predictions()
        self.run_evaluation()


def parse_args():
    parser = argparse.ArgumentParser(description="main")
    parser.add_argument("--model_name", type=str, required=True, help="Set the model name, e.g. `blip_vqa`")
    parser.add_argument("--dataset_name", type=str, help="Set the dataset name, e.g. `winoground`", required=True)
    parser.add_argument(
        "--prompt_name", default="ensemble", type=str, help="Set the prompt name, e.g. `does_caption_match_wo_answer`"
    )
    parser.add_argument(
        "--task_type",
        default="open_ended",
        choices=["multiple_choice", "open_ended"],
        type=str,
        help="Set the task type, e.g. `multiple_choice` or `open_ended`",
    )
    parser.add_argument(
        "--vqa_format",
        type=str,
        default="standard_vqa",
        choices=["standard_vqa", "caption_vqa", "cot_vqa"],
    )
    parser.add_argument(
        "--gen_model_name",
        type=str,
        default=None,
        choices=[
            "gpt3",
            "chatgpt",
            "flan_t5",
            "promptcap",
            "blip2_opt27b",
            "blip2_opt67b",
            "blip2_flant5xl",
            "blip2_flant5xxl",
            "kosmos2",
        ],
        help="choose generator model type",
    )
    parser.add_argument(
        "--split_name",
        type=str,
        default="val",
        choices=["val", "testdev_bal"],
        help="choose generator model type",
    )
    parser.add_argument("--vicuna_ans_parser", action="store_true", help="Set to True to run vicuna eval.")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(vars(args))
    ensemble = EnsemblePredictions(args)
    ensemble.process()
