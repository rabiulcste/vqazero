import argparse
import json
import os
import re
import string

from tqdm import tqdm
from fuzzywuzzy import fuzz
from fuzzywuzzy import fuzz

from carets.carets.dataset import CaretsDataset
from dataset_zoo.custom_dataset import VQADataset
from modeling_utils import save_to_json
from utils.okvqa_utils import VQAAnswerProcessor, postprocess_batch_vqa_generation_blip2
from utils.config import SCRATCH_DIR
from utils.logger import Logger
from utils.vqa import VQA
from utils.vqaEval import VQAEval

logger = Logger(__name__)


def normalize_string(input_string):
    input_string = input_string.lower()
    input_string = input_string.replace("\n", "")  # remove newline characters
    input_string = input_string.strip()  # remove leading and trailing whitespaces
    input_string = input_string.strip(string.punctuation)  # remove punctuation
    input_string = input_string.strip()  # remove leading and trailing whitespaces
    return input_string


def eval_vqa(args, output_dir: str):
    # create vqa object and vqaRes object
    res_file = os.path.join(output_dir, "annotations+vqa_answers.json")
    vqa = VQA(dataset_name=args.dataset_name)
    vqaRes = vqa.loadRes(resFile=res_file)

    # create vqaEval object by taking vqa and vqaRes
    vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2

    # evaluate results
    """
    If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
    By default it uses all the question ids in annotation file
    """
    vqaEval.evaluate()

    # print accuracies
    print("\n")
    print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy["overall"]))
    print("Per Question Type Accuracy is the following:")
    for quesType in vqaEval.accuracy["perQuestionType"]:
        print("%s : %.02f" % (quesType, vqaEval.accuracy["perQuestionType"][quesType]))
    print("\n")
    print("Per Answer Type Accuracy is the following:")
    for ansType in vqaEval.accuracy["perAnswerType"]:
        print("%s : %.02f" % (ansType, vqaEval.accuracy["perAnswerType"][ansType]))

    data = {
        "dataset_name": args.dataset_name,
        "dataset_name": args.dataset_name,
        "prompt_name": args.prompt_name,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": len(vqa.dataset["annotations"]),
        "vqa_accuracy": vqaEval.accuracy["overall"],
        "fuzzy_accuracy": vqaEval.accuracy["fuzzyOverall"],
        "fuzzy_accuracy": vqaEval.accuracy["fuzzyOverall"],
        "per_question_type_accuracy": vqaEval.accuracy["perQuestionType"],
        "per_answer_type_accuracy": vqaEval.accuracy["perAnswerType"],
    }

    json_file = os.path.join(output_dir, "result_meta.json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")


def eval_gqa(args, output_dir: str):
    res_file = os.path.join(output_dir, "predictions.json")
    with open(res_file, "r") as f:
        results = json.load(f)
    total_correct = 0
    total_fuzz_correct = 0
    for question_id, result in results.items():
        if result["answer"] == result["prediction"]:
            total_correct += 1
        fuzz_score = fuzz.token_set_ratio(result["prediction"], result["answer"], force_ascii=False)
        if fuzz_score > 80:
            total_fuzz_correct += 1

    overall_acc = total_correct / len(results)
    overall_fuzz_acc = total_fuzz_correct / len(results)
    data = {
        "dataset_name": "gqa",
        "prompt_name": args.prompt_name,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": len(results),
        "vqa_accuracy": overall_acc,
        "fuzzy_accuracy": overall_fuzz_acc,
    }
    json_file = os.path.join(output_dir, "result_meta.json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")


def eval_carets(args, output_dir: str, predictions_all_splits):
    dataset = CaretsDataset("./carets/configs/default.yml")
    accuracy_dict = dict()
    comprehensive_accuracy_dict = dict()
    num_examples_dict = dict()
    for test_name, split in dataset.splits:
        print(split)

        predictions = predictions_all_splits[test_name]
        predictions = {qid: predictions[qid]["prediction"] for qid in predictions}

        accuracy = split.total_accuracy(predictions)
        consistency = split.evaluate(predictions)
        comprehensive_accuracy = split.comprehensive_accuracy(predictions)
        eval_type = split.eval_type
        print(
            f"{test_name.ljust(24)}: accuracy: {accuracy:.3f}, {eval_type.ljust(24)}:"
            + f" {consistency:.3f}, comprehensive_accuracy: {comprehensive_accuracy:.3f}"
        )
        accuracy_dict[test_name] = accuracy
        comprehensive_accuracy_dict[test_name] = comprehensive_accuracy
        num_examples_dict[test_name] = len(split.questions)

    data = {
        "dataset_name": args.dataset_name,
        "prompt_name": args.prompt_name,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": num_examples_dict,
        "vqa_accuracy": accuracy_dict,
        "comprehensive_accuracy": comprehensive_accuracy_dict,
    }

    logger.info(f"VQA Accuracy: {accuracy_dict}")
    json_file = os.path.join(output_dir, "result_meta.json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")


# https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py
def eval_visual7w(args, output_dir: str, preds, multiple_choice=False, strict=False):
    preds = {int(qid): preds[qid]["raw_prediction"] for qid in preds}
    v7w_dataset = VQADataset(args, args.dataset_name)
    dataset = v7w_dataset.get_image_qa_multiple_choice_dataset()
    if isinstance(dataset, list):
        dataset = {dataset[i]["question_id"]: dataset[i] for i in range(len(dataset))}
    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids), "Some questions are missing predictions"

    acc = []
    for q in preds.keys():
        pred = preds[q]
        choices = dataset[q]["mc"]
        direct_answers = [dataset[q]["answer"]]  # just one answer

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, "Prediction must be a valid choice"
            correct_choice_idx = dataset[q]["mc_selection"]
            acc.append(float(normalize_string(pred) == normalize_string(dataset[q]["answer"])))
        ## Direct Answer setting
        else:
            num_match = sum([normalize_string(pred) == normalize_string(da) for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0)
            acc.append(vqa_acc)

    acc = sum(acc) / len(acc) * 100

    data = {
        "dataset_name": args.dataset_name,
        "prompt_name": args.prompt_name,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": len(dataset),
        "vqa_accuracy": acc,
    }

    logger.info(f"VQA Accuracy: {acc}")
    json_file = os.path.join(output_dir, "result_meta.json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")


def _load_aokvqa(aokvqa_dir, split, version="v1p0"):
    assert split in ["train", "val", "test", "test_w_ans"]
    dataset = json.load(open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")))
    return dataset


def eval_aokvqa(args, output_dir: str, preds, multiple_choice=False, strict=False):
    preds = {qid: preds[qid]["prediction"] for qid in preds}  # flamingo_processed_prediction
    dataset = _load_aokvqa(os.path.join(SCRATCH_DIR, "datasets", "aokvqa"), "val")
    if isinstance(dataset, list):
        dataset = {dataset[i]["question_id"]: dataset[i] for i in range(len(dataset))}

    if multiple_choice is False:
        dataset = {k: v for k, v in dataset.items() if v["difficult_direct_answer"] is False}

    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids)

    # dataset = q_id (str) : dataset element (dict)
    # preds = q_id (str) : prediction (str)

    acc = []

    for q in tqdm(dataset.keys(), desc="Evaluating"):
        if q not in preds.keys():
            acc.append(0.0)
            continue

        pred = preds[q]
        choices = dataset[q]["choices"]
        direct_answers = dataset[q]["direct_answers"]

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, "Prediction must be a valid choice"
            correct_choice_idx = dataset[q]["correct_choice_idx"]
            cur_acc = float(normalize_string(pred) == normalize_string(choices[correct_choice_idx]))
            acc.append(cur_acc)
        ## Direct Answer setting
        else:
            gtAcc = []
            direct_answers = postprocess_batch_vqa_generation_blip2(args.dataset_name, direct_answers)
            direct_answers = [{"answer": da, "answer_id": idx} for idx, da in enumerate(direct_answers)]
            for gtAnsDatum in direct_answers:
                otherGTAns = [item for item in direct_answers if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item["answer"] == pred]
                curr_acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(curr_acc)
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            # print("avgGTAcc", avgGTAcc, direct_answers, pred)
            acc.append(avgGTAcc)
            """
            direct_answers = [aokvqa_proc.process_aokvqa(da) for da in direct_answers]
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0) 
            acc.append(vqa_acc)
            """
    print("acc", sum(acc))
    acc = sum(acc) / len(acc) * 100
    data = {
        "dataset_name": args.dataset_name,
        "prompt_name": args.prompt_name,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": len(dataset),
        "vqa_accuracy": acc,
    }

    logger.info(f"VQA Accuracy: {acc}")
    json_file = os.path.join(output_dir, "result_meta.json")
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")


def eval_winoground(args, output_dir: str, predictions, template_expr=""):
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
    for result in predictions:
        _result = {}
        for key, sentence in result.items():
            if key not in ["c0_i0", "c0_i1", "c1_i0", "c1_i1"]:
                continue
            match = re.search(r"^\w+", str(sentence))  # find the first word that matches the pattern
            if match:
                _result[key] = match.group().lower()
            else:
                _result[key] = sentence

        group_correct_count += 1 if group_correct(_result) else 0
        common_correct_count += 1 if common_correct(_result) else 0
        uncommon_correct_count += 1 if uncommon_correct(_result) else 0

    denominator = len(predictions)
    logger.info(f"group score: {group_correct_count * 100 / denominator}")
    logger.info(f"common score: {common_correct_count * 100 / denominator}")
    logger.info(f"uncommon score: {uncommon_correct_count * 100 / denominator}")

    # Save the results as a JSON file
    data = {
        "dataset_name": "winoground",
        "prompt_name": args.prompt_name,
        "template_expression": template_expr,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": len(predictions),
        "group_score": group_correct_count * 100 / denominator,
        "common_score": common_correct_count * 100 / denominator,
        "uncommon_score": uncommon_correct_count * 100 / denominator,
    }
    json_path = os.path.join(output_dir, "result_meta.json")

    save_to_json(json_path, data)
    logger.info(f"Saved result information to {json_path}")
