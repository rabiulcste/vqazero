
import torch
import json
import os
import random
import re

import torch
from bert_score import score
from rouge import Rouge

from utils.logger import Logger
from utils.okvqa_utils import extract_answer_from_cot

# Set up logger
logger = Logger(__name__)

root_dir = "/home/mila/r/rabiul.awal/scratch/decompose-composition/"


def load_data():
    ground_truth_path = os.path.join(root_dir, "datasets/aokvqa/aokvqa_v1p0_val.json")
    with open(ground_truth_path, "r") as f:
        ground_truth = json.load(f)

    human_explanations = {}
    for question_obj in ground_truth:
        question_id = question_obj["question_id"]
        all_rationales = question_obj["rationales"]
        # rationale = all_rationales[random.randint(0, len(all_rationales)-1)]
        rationale = max(all_rationales, key=lambda x: len(x))
        human_explanations[question_id] = rationale

    lm_explanations_path = os.path.join(
        root_dir,
        # "output/aokvqa/blip2_t5_flant5xl/basic_qa/prefix_instruct_rationale/open_ended/predictions.json",
        # "output/aokvqa/blip2_t5_flant5xl/cot_qa/prefix_rationale_before_answering_knn/prefix_rationale_before_answering/open_ended/predictions.json"
        # "output/aokvqa/blip2_t5_flant5xxl/basic_qa/prefix_rationale_before_answering/open_ended/predictions.json"
        "output/aokvqa/blip2_t5_flant5xxl/cot_qa/prefix_instruct_rationale_knn/prefix_instruct_rationale/open_ended/predictions.json"
    )
    with open(lm_explanations_path, "r") as f:
        lm_explanations = json.load(f)

    human_explanations_list = []
    lm_explanations_list = []
    question_list = []
    answer_list = []
    for question_id in human_explanations:
        if "predictions" in lm_explanations_path:
            curr_lm_explanation = lm_explanations[question_id]["raw_prediction"]
        else:
            curr_lm_explanation = lm_explanations[question_id]
        curr_question = lm_explanations[question_id]["question"]
        curr_answer  = lm_explanations[question_id]["answer"]
        curr_lm_explanation = remove_answer(curr_lm_explanation)
        if not curr_lm_explanation:
            logger.warning(f"No explanation generated for question {question_id}")
            continue
        lm_explanations_list.append(curr_lm_explanation)
        human_explanations_list.append(human_explanations[question_id])
        question_list.append(curr_question)
        answer_list.append(curr_answer)

    return human_explanations_list, lm_explanations_list, question_list, answer_list


def remove_answer(rationale):
    rationale = rationale.replace("To answer the above question, the relevant sentence is:", "").strip()
    if not rationale:
        print("Empty rationale")
    extracted_answer = extract_answer_from_cot(rationale)
    if extracted_answer != rationale:
        # type 1
        # match = re.search(r'\. (?=[A-Z])', rationale)
        # if match:
        #     rationale = rationale[:match.end()-1]

        # type 2
        answer_start_index = rationale.rfind(extracted_answer)
        last_period_index = rationale[:answer_start_index].rfind(".")
        rationale = rationale[: last_period_index + 1].strip()
    return rationale


def calculate_average_bertscore(
    human_explanations,
    lm_explanations,
    lang="en",
    save_pairwise_metrics=True,
    output_path="pairwise_bertscore_metrics.json",
):
    # Ensure that the number of explanations in both lists is the same
    assert len(human_explanations) == len(lm_explanations)

    # Calculate BERTScores
    P, R, F1 = score(
        lm_explanations,
        human_explanations,
        lang=lang,
        verbose=True,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    # Calculate average BERTScores
    avg_P = P.mean().item()
    avg_R = R.mean().item()
    avg_F1 = F1.mean().item()

    if save_pairwise_metrics:
        pairwise_metrics = []
        for i in range(len(human_explanations)):
            pairwise_metrics.append(
                {
                    "ground_truth": human_explanations[i],
                    "predicted_text": lm_explanations[i],
                    "precision": P[i].item(),
                    "recall": R[i].item(),
                    "f1": F1[i].item(),
                }
            )

        with open(output_path, "w") as f:
            json.dump(pairwise_metrics, f, indent=2)

        logger.info(f"Saved pairwise BERTScore metrics to {output_path}")

    return avg_P, avg_R, avg_F1


def calculate_average_rouge_score(
    human_explanations, lm_explanations, save_pairwise_metrics=True, output_path="pairwise_rouge_metrics.json"
):
    # Ensure that the number of explanations in both lists is the same
    assert len(human_explanations) == len(lm_explanations)

    # Initialize Rouge object
    rouge = Rouge()

    # Calculate ROUGE scores
    rouge_scores = rouge.get_scores(lm_explanations, human_explanations, avg=True)

    if save_pairwise_metrics:
        pairwise_metrics = []
        for i in range(len(human_explanations)):
            scores = rouge.get_scores(lm_explanations[i], human_explanations[i])[0]
            pairwise_metrics.append(
                {
                    "ground_truth": human_explanations[i],
                    "predicted_text": lm_explanations[i],
                    "rouge_1": scores["rouge-1"],
                    "rouge_2": scores["rouge-2"],
                    "rouge_l": scores["rouge-l"],
                }
            )

        with open(output_path, "w") as f:
            json.dump(pairwise_metrics, f, indent=2)

        logger.info(f"Saved pairwise ROUGE metrics to {output_path}")

    return rouge_scores


human_explanations, lm_explanations, question_list, answer_list = load_data()

for q, a, gp, lp in zip(question_list, answer_list, human_explanations, lm_explanations):
    print(f"{q} = {a} | {gp} | {lp}")




# # Calculate the average BERTScore metrics
# bertscore_output_path = os.path.join(root_dir, "output/aokvqa/report/explanations_eval/bertscore.json")
# avg_P, avg_R, avg_F1 = calculate_average_bertscore(
#     human_explanations, lm_explanations, lang="en", output_path=bertscore_output_path
# )

# # Print the average BERTScores
# logger.info(f"Average BERTScore (Precision): {avg_P:.4f}")
# logger.info(f"Average BERTScore (Recall): {avg_R:.4f}")
# logger.info(f"Average BERTScore (F1): {avg_F1:.4f}")

# # Calculate the average ROUGE scores
# rouge_output_path = os.path.join(root_dir, "output/aokvqa/report/explanations_eval/rouge.json")
# rouge_scores = calculate_average_rouge_score(human_explanations, lm_explanations, output_path=rouge_output_path)

# # Print the average ROUGE scores
# logger.info(f"Average ROUGE-1 (Precision): {rouge_scores['rouge-1']['p']:.4f}")
# logger.info(f"Average ROUGE-1 (Recall): {rouge_scores['rouge-1']['r']:.4f}")
# logger.info(f"Average ROUGE-1 (F1): {rouge_scores['rouge-1']['f']:.4f}")
# logger.info(f"Average ROUGE-2 (Precision): {rouge_scores['rouge-2']['p']:.4f}")
# logger.info(f"Average ROUGE-2 (Recall): {rouge_scores['rouge-2']['r']:.4f}")
# logger.info(f"Average ROUGE-2 (F1): {rouge_scores['rouge-2']['f']:.4f}")
# logger.info(f"Average ROUGE-L (Precision): {rouge_scores['rouge-l']['p']:.4f}")
# logger.info(f"Average ROUGE-L (Recall): {rouge_scores['rouge-l']['r']:.4f}")
# logger.info(f"Average ROUGE-L (F1): {rouge_scores['rouge-l']['f']:.4f}")
