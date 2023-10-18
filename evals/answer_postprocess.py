import json
import re
import string
from collections import Counter
from functools import reduce
from typing import List, Union

dont_change_list = ["left"]


manualMap = {
    "none": "0",
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
}
articles = ["a", "an", "the"]


def process_digit_article(inText):
    outText = []
    tempText = inText.lower().split()
    for word in tempText:
        word = manualMap.setdefault(word, word)
        if word not in articles:
            outText.append(word)
        else:
            pass
    outText = " ".join(outText)

    return outText


def clean_last_word(s):  # sun10 -> sun; for vicuna llm answer extraction case only
    return re.sub(r"(?<=[a-zA-Z])\d+(?=\s*$)", "", s)


def normalize_string(input_string):
    input_string = input_string.lower()
    input_string = input_string.replace("\n", "")  # remove newline characters
    input_string = input_string.strip()  # remove leading and trailing whitespaces
    input_string = input_string.strip(string.punctuation)  # remove punctuation
    input_string = input_string.strip()  # remove leading and trailing whitespaces

    return input_string


class VQAAnswerProcessor:
    """Processor for VQA answer strings."""

    def __init__(self):
        import spacy

        spacy.prefer_gpu()
        self.lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def process(self, dataset_name: str, answers: List[str], questions=None):
        answers = [self.strip_prefix(answer) for answer in answers]

        if questions is not None:
            if dataset_name == "aokvqa":
                answers = [self.extract_text_before_or(answer) for answer in answers]

            if dataset_name in ["okvqa", "gqa"]:
                answers = self.strip_matching_noun_suffixes(answers, questions)

            if dataset_name in ["okvqa"] and questions:
                answers = self.lemmatize_answer_texts(answers)

        answers = [process_digit_article(pred) for pred in answers]
        answers = [input_string.strip(string.punctuation) for input_string in answers]  # remove punctuation
        answers = [answer.strip() for answer in answers]
        answers = [answer.lower() for answer in answers]

        return answers

    def lemmatize_text(self, input_string: str):
        doc = self.lemmatizer(input_string)
        words = []
        for token in doc:
            if token.pos_ in ["NOUN", "VERB"]:
                # Preserve the original whitespace following the token after lemmatization
                words.append(token.lemma_ + token.whitespace_)
            else:
                # Preserve the original token along with the whitespace that follows it
                words.append(token.text_with_ws)
        lemmatized_answer = "".join(words)

        return lemmatized_answer

    def lemmatize_answer_texts(self, answers: List[str]):
        lemmatized_answers = []
        dont_change_set = set(dont_change_list)
        for doc in self.lemmatizer.pipe(answers, n_process=-1):
            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"] and token.text not in dont_change_set:
                    # Preserve the original whitespace following the token after lemmatization
                    words.append(token.lemma_ + token.whitespace_)
                else:
                    # Preserve the original token along with the whitespace that follows it
                    words.append(token.text_with_ws)
            lemmatized_answer = "".join(words)
            lemmatized_answers.append(lemmatized_answer)

        return lemmatized_answers

    def strip_matching_noun_suffixes(self, answers, questions):
        """Remove answer suffix if it is a noun found in the corresponding question."""
        question_nouns = [
            {token.text for token in doc if token.pos_ == "NOUN"}
            for doc in self.lemmatizer.pipe(questions, n_process=-1)
        ]

        processed_answers = []
        for answer, nouns in zip(answers, question_nouns):
            answer_words = answer.split()
            if len(answer_words) < 2 or len(answer_words) > 3:
                processed_answers.append(answer)
                continue

            last_word = answer_words[-1]
            if last_word in nouns:
                processed_answer = " ".join(answer_words[:-1])
            else:
                processed_answer = answer
            processed_answers.append(processed_answer)

        return processed_answers

    @staticmethod
    def extract_text_before_or(text: str) -> str:
        """
        If the word 'or' exists in a string, return the part before 'or'.
        Otherwise, return an empty string.

        Args:
        - text (str): The input string to check.

        Returns:
        - str: The part of the string before 'or' if 'or' exists, otherwise an empty string.
        """
        match = re.search(r"(.+?)\s+or\s+", text, re.IGNORECASE)
        return match.group(1).strip() if match else text

    @staticmethod
    def strip_punctuation(word):
        # Define a regular expression pattern that matches punctuation marks at the beginning or end of the word
        pattern = r"^[\W_]+|[\W_]+$"

        # Use the sub() method of the re module to remove the punctuation marks
        return re.sub(pattern, "", word)

    @staticmethod
    def strip_prefix(answer):
        """Remove a prefix word from an answer string if it matches a list of known prefixes."""
        answer = str(answer)  # making sure it is a string
        prefix_list = ["to", "in", "at"]
        answer = answer.split()
        if len(answer) > 1 and answer[0] in prefix_list:
            answer = answer[1:]

        return " ".join(answer)

    @staticmethod
    def strip_articles(answer):
        """Remove articles from an answer string."""
        pattern = r"^(a|an|the)\s+|\s+(a|an|the)\b"
        answer = re.sub(pattern, "", answer)
        return answer


answer_processor = VQAAnswerProcessor()


# def postprocess_vqa_answers(dataset_name: str, predictions: Union[List[str], List[List[str]]], questions=None) -> str:
#     return answer_processor.process(dataset_name, predictions, questions)


def postprocess_vqa_answers(
    dataset_name: str, predictions: Union[List[str], List[List[str]]], questions=None
) -> Union[str, List[str]]:
    # Determine if predictions is a 2D list (list of lists)
    is_2d = isinstance(predictions[0], list) if predictions else False

    # Flatten predictions if it's 2D for efficient processing
    if is_2d:
        original_shape = [len(sublist) for sublist in predictions]
        predictions = [ans for sublist in predictions for ans in sublist]

        # Extend or repeat questions to match the size of flattened predictions
        if questions is not None:
            questions = [q for q, shape in zip(questions, original_shape) for _ in range(shape)]

    # Process the predictions
    processed_predictions = answer_processor.process(dataset_name, predictions, questions)

    # If predictions were originally 2D, reshape them back into 2D
    if is_2d:
        start_idx = 0
        reshaped_predictions = []
        for shape in original_shape:
            end_idx = start_idx + shape
            reshaped_predictions.append(processed_predictions[start_idx:end_idx])
            start_idx = end_idx
        processed_predictions = reshaped_predictions

    return processed_predictions


def extract_rationale_per_question(answers, sz):
    num_questions = len(answers) // sz
    results = []
    for i in range(num_questions):
        question_answers = answers[i * sz : (i + 1) * sz]
        results.append(question_answers)
    return results


def extract_answer_from_cot(sentence):
    patterns = [
        r'final answer is ["\']?([A-Za-z0-9\s\'\-\.\(\)]+)["\']?',
        r"final answer is ([A-Za-z0-9\s\'\-\.\(\)]+)\.",
        r"the final answer is ([A-Za-z0-9\s\'\-\.\(\)]+)\.",
        r"Therefore, the final answer is ([A-Za-z0-9\s\'\-\.\(\)]+)\.",
        r"final answer is ([A-Za-z0-9\s\'\-\.\(\)]+)",
        r"The final answer: ([A-Za-z0-9\s\'\-\.\(\)]+)",
        r"The answer: ([A-Za-z0-9\s\'\-\.\(\)]+)\.",
        r"The answer: ([A-Za-z0-9\s\'\-\.\(\)]+)\Z",
        r"The answer is ([A-Za-z0-9\s\'\-\.\(\)]+)\.",
        r"The answer is ([A-Za-z0-9\s\'\-\.\(\)]+)\Z",
    ]

    for pattern in patterns:
        match = re.search(pattern, sentence, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            answer = re.sub(r"^\W+|\W+$", "", answer)  # remove leading/trailing punctuation
            return answer

    return sentence


def majority_vote_with_indices_2d(answers: List[List[str]]):
    results = []
    indices = []
    for question_answers in answers:
        counts = {}
        for idx, ans in enumerate(question_answers):
            if not ans:
                continue
            if ans in counts:
                counts[ans]["count"] += 1
                counts[ans]["indices"].append(idx)
            else:
                counts[ans] = {"count": 1, "indices": [idx]}
        max_count = max(counts.values(), key=lambda x: x["count"])["count"] if counts else 0
        max_ans = [(k, v["indices"]) for k, v in counts.items() if v["count"] == max_count]
        if max_ans:
            results.append(max_ans[0][0])
            indices.append(max_ans[0][1][0])  # Directly append the index of the answer within question_answers
        else:
            results.append("")
            indices.append(0)  # Append 0 or another default value if no answer is found

    return results, indices


def majority_vote(answers: List[str]):
    """
    Find the most common answer in a list of answers.

    Args:
    - answers (List[str]): A list of answer strings.

    Returns:
    - str: The most common answer in the input list.
    """
    # Count the occurrences of each answer
    answer_counts = Counter(answers)

    # Find the most common answer
    most_common_answer, _ = answer_counts.most_common(1)[0] if answer_counts else ("", 0)

    return most_common_answer


def postprocess_cleanup_answer(args, predictions, logger):
    qids = list(predictions.keys())
    batch = [predictions[qid] for qid in qids]
    output = [data["generated_output"] for data in batch]

    if args.self_consistency:  # only for llava and blip2 models
        logger.info(f"RAW PREDICTION: {json.dumps(output, indent=2)}")
        if "rationale" in args.prompt_name:
            extracted_answers = [[extract_answer_from_cot(i) for i in o_group] for o_group in output]
            for i, data in enumerate(batch):
                data["reasoning_paths"] = output[i]
                data["reasoning_answers"] = extracted_answers[i]
        else:
            extracted_answers = list(output)

        processed_answers = postprocess_vqa_answers(args.dataset_name, extracted_answers)
        majority_answers, indices = majority_vote_with_indices_2d(processed_answers)

        for i, data in enumerate(batch):
            data["raw_prediction"] = output[i][indices[i]]
            data["prediction"] = majority_answers[i]

    else:
        questions = [data["question"] for data in batch]
        partitions = [". ", "Q: ", "Question: ", "Long Answer: "]
        if args.model_name in [
            "kosmos2",
            "opt27b",
            "opt67b",
            "open_flamingo_mpt",
            "open_flamingo_redpajama",
            "open_flamingo_redpajama_instruct",
        ]:
            questions = [re.sub("^<grounding> ", "", q) for q in questions]
            output = [re.split(r"[\n.]", o)[0] for o in output]
            output = [reduce(lambda txt, p: txt.partition(p)[0], partitions, text) for text in output]
            output = [o.strip() for o in output]

        for i, data in enumerate(batch):
            data["raw_prediction"] = output[i]

        if "rationale" in args.prompt_name and "iterative" not in args.prompt_name:
            output = [extract_answer_from_cot(prediction) for prediction in output]

        if args.dataset_name in ["aokvqa", "visual7w"] and args.task_type == "multiple_choice":
            for i, data in enumerate(batch):
                data["prediction"] = output[i]
        else:
            cleaned_results = postprocess_vqa_answers(
                args.dataset_name, output, questions
            )  # answer batch processing for blip2
            for i, data in enumerate(batch):
                data["prediction"] = cleaned_results[i]

    return predictions
