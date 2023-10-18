import json
import os
import re
from datetime import datetime

from thefuzz import fuzz
from tqdm import tqdm

from dataset_zoo.common import load_visual7w_dataset_from_json
from evals.answer_postprocess import normalize_string, postprocess_vqa_answers
from evals.vqa import VQA
from evals.vqaEval import VQAEval
from utils.config import VQA_DATASET_DIR
from utils.logger import Logger
from vqa_zero.inference_utils import save_to_json

logger = Logger(__name__)


def eval_vqa(args, output_dir: str, vicuna_ans_parser=False):
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
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    fn_suffx = "result_meta_vicuna.json" if vicuna_ans_parser else "result_meta.json"
    json_file = os.path.join(output_dir, fn_suffx)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")


def eval_gqa(args, output_dir: str, vicuna_ans_parser: bool = False, results: dict = None):
    if results is None:
        res_file = os.path.join(output_dir, "predictions.json")
        with open(res_file, "r") as f:
            results = json.load(f)

    total_correct = 0
    total_fuzz_correct = 0
    wrong_predictions = []
    for question_id, result in results.items():
        if result["answer"] == result["prediction"]:
            total_correct += 1
        else:
            wrong_predictions.append(result)

        fuzz_score = fuzz.token_set_ratio(result["prediction"], result["answer"], force_ascii=False)
        if fuzz_score > 80:
            total_fuzz_correct += 1

    overall_acc = total_correct * 100 / len(results)
    print(f"Overall accuracy: {overall_acc}")
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
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    fn_suffx = "result_meta_vicuna.json" if vicuna_ans_parser else "result_meta.json"
    json_file = os.path.join(output_dir, fn_suffx)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")

    wrong_predictions_file = os.path.join(output_dir, "wrong_predictions.json")
    with open(wrong_predictions_file, "w") as f:
        json.dump(wrong_predictions, f, indent=4)
    logger.info(f"Saved incorrect predictions to {wrong_predictions_file}")


# https://github.com/allenai/aokvqa/blob/main/evaluation/eval_predictions.py
def eval_visual7w(args, output_dir: str, predictions, multiple_choice=False, strict=False, vicuna_ans_parser=False):
    preds = {str(qid): predictions[qid]["prediction"] for qid in predictions}
    dataset = load_visual7w_dataset_from_json(args.dataset_name, "val")
    if isinstance(dataset, list):
        dataset = {str(dataset[i]["question_id"]): dataset[i] for i in range(len(dataset))}
    if strict:
        dataset_qids = set(dataset.keys())
        preds_qids = set(preds.keys())
        assert dataset_qids.issubset(preds_qids), "Some questions are missing predictions"

    acc = []
    wrong_predictions = []
    for q in preds.keys():
        pred = preds[q]
        choices = dataset[q]["choice"]
        direct_answers = [dataset[q]["answer"]]  # just one answer

        ## Multiple Choice setting
        if multiple_choice:
            if strict:
                assert pred in choices, "Prediction must be a valid choice"
            cur_acc = float(normalize_string(pred) == normalize_string(dataset[q]["answer"]))
            acc.append(cur_acc)
            if cur_acc == 0:
                curr_info_dict = predictions.get(q, predictions.get(str(q)))
                wrong_predictions.append(curr_info_dict)

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
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info(f"VQA Accuracy: {acc}")
    fn_suffx = "result_meta_vicuna.json" if vicuna_ans_parser else "result_meta.json"
    json_file = os.path.join(output_dir, fn_suffx)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")

    wrong_predictions_file = os.path.join(output_dir, "wrong_predictions.json")
    with open(wrong_predictions_file, "w") as f:
        json.dump(wrong_predictions, f, indent=4)
    logger.info(f"Saved incorrect predictions to {wrong_predictions_file}")


def _load_aokvqa(aokvqa_dir, split, version="v1p0"):
    assert split in ["train", "val", "test", "test_w_ans"]
    dataset = json.load(open(os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")))
    return dataset


def eval_aokvqa(args, output_dir: str, predictions, multiple_choice=False, strict=False, vicuna_ans_parser=False):
    preds = {qid: predictions[qid]["prediction"] for qid in predictions}  # flamingo_processed_prediction
    dataset = _load_aokvqa(os.path.join(VQA_DATASET_DIR, "datasets", "aokvqa"), args.split)
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
    wrong_predictions = []
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
            if cur_acc == 0:
                curr_info_dict = predictions.get(q, predictions.get(str(q)))
                wrong_predictions.append(curr_info_dict)

        ## Direct Answer setting
        else:
            gtAcc = []
            direct_answers = postprocess_vqa_answers(args.dataset_name, direct_answers)
            direct_answers = [{"answer": da, "answer_id": idx} for idx, da in enumerate(direct_answers)]

            for gtAnsDatum in direct_answers:
                otherGTAns = [item for item in direct_answers if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item["answer"] == pred]
                curr_acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(curr_acc)
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            acc.append(avgGTAcc)

            if avgGTAcc == 0:
                wrong_predictions.append(predictions[q])

            """
            direct_answers = [aokvqa_proc.process_aokvqa(da) for da in direct_answers]
            num_match = sum([pred == da for da in direct_answers])
            vqa_acc = min(1.0, num_match / 3.0) 
            acc.append(vqa_acc)
            """
    acc = sum(acc) / len(acc) * 100

    data = {
        "dataset_name": args.dataset_name,
        "prompt_name": args.prompt_name,
        "model_name": args.model_name,
        "gen_model_name": args.gen_model_name,
        "vqa_format": args.vqa_format,
        "num_examples": len(dataset),
        "vqa_accuracy": acc,
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.info(f"VQA Accuracy: {acc}")
    fn_suffx = "result_meta_vicuna.json" if vicuna_ans_parser else "result_meta.json"
    json_file = os.path.join(output_dir, fn_suffx)
    with open(json_file, "w") as f:
        json.dump(data, f, indent=4)
    logger.info(f"Saved result meta to {json_file}")

    wrong_predictions_file = os.path.join(output_dir, "wrong_predictions.json")
    with open(wrong_predictions_file, "w") as f:
        json.dump(wrong_predictions, f, indent=4)
    logger.info(f"Saved incorrect predictions to {wrong_predictions_file}")


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
    for data in predictions:
        result = data["scores"]
        group_correct_count += 1 if group_correct(result) else 0
        common_correct_count += 1 if common_correct(result) else 0
        uncommon_correct_count += 1 if uncommon_correct(result) else 0

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
        "eval_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    json_path = os.path.join(output_dir, "result_meta.json")
    save_to_json(json_path, data)
