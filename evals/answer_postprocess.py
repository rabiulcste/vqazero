import json
import re
import string
from functools import reduce
from typing import List

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

    def process(self, dataset_name, answers: List[str], questions=None):
        answers = [self.fix_prefix(answer) for answer in answers]

        if questions is not None:
            if dataset_name == "aokvqa":
                answers = [self.extract_before_or(answer) for answer in answers]

            if dataset_name in ["okvqa", "gqa"]:
                answers = self.remove_matching_noun_suffix_from_answers(answers, questions)

            if dataset_name in ["okvqa"] and questions:
                answers = self.lemmatize(answers)

        answers = [process_digit_article(pred) for pred in answers]
        answers = [input_string.strip(string.punctuation) for input_string in answers]  # remove punctuation
        answers = [answer.strip() for answer in answers]
        answers = [answer.lower() for answer in answers]

        return answers

    def lemmatize_single(self, input_string):
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

    def lemmatize(self, answers):
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

    def remove_matching_noun_suffix_from_answers(self, answers, questions):
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
    def extract_before_or(text: str) -> str:
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
    def remove_punctuation(word):
        # Define a regular expression pattern that matches punctuation marks at the beginning or end of the word
        pattern = r"^[\W_]+|[\W_]+$"

        # Use the sub() method of the re module to remove the punctuation marks
        return re.sub(pattern, "", word)

    @staticmethod
    def fix_prefix(answer):
        """Remove a prefix word from an answer string if it matches a list of known prefixes."""
        answer = str(answer)  # making sure it is a string
        prefix_list = ["to", "in", "at"]
        answer = answer.split()
        if len(answer) > 1 and answer[0] in prefix_list:
            answer = answer[1:]

        return " ".join(answer)

    @staticmethod
    def remove_articles(answer):
        """Remove articles from an answer string."""
        pattern = r"^(a|an|the)\s+|\s+(a|an|the)\b"
        answer = re.sub(pattern, "", answer)
        return answer


answer_processor = VQAAnswerProcessor()


def postprocess_vqa_answers(dataset_name: str, predictions: List[str], questions=None) -> str:
    return answer_processor.process(dataset_name, predictions, questions)


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


def majority_vote_with_indices(answers, sz):
    num_questions = len(answers) // sz
    results = []
    indices = []
    for i in range(num_questions):
        question_answers = answers[i * sz : (i + 1) * sz]
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
            indices.append(i * sz + max_ans[0][1][0])
        else:
            results.append("")
            indices.append(i * sz)
    return results, indices


def postprocess_cleanup_answer(args, predictions, logger):
    qids = list(predictions.keys())
    batch = [predictions[qid] for qid in qids]
    output = [data["generated_output"] for data in batch]

    if args.self_consistency:
        logger.info(f"RAW PREDICTION: {json.dumps(output, indent=2)}")
        if "rationale" in args.prompt_name:
            batch["reasoning_paths"] = extract_rationale_per_question(output, args.num_captions)
            extracted_answers = [extract_answer_from_cot(prediction) for prediction in output]
            batch["reasoning_answers"] = extract_rationale_per_question(extracted_answers, args.num_captions)
        else:
            extracted_answers = list(output)
        processed_answers = postprocess_vqa_answers(args.dataset_name, extracted_answers)
        majority_answers, indices = majority_vote_with_indices(processed_answers, args.num_captions)
        batch["raw_prediction"] = [output[i] for i in indices]
        batch["prediction"] = majority_answers
        output = batch["raw_prediction"]

    else:
        questions = [data["question"] for data in batch]

        if "rationale" in args.prompt_name and "iterative" not in args.prompt_name:
            output = [extract_answer_from_cot(prediction) for prediction in output]

        partitions = [". ", "Q: ", "Question: "]
        if args.model_name in ["kosmos2", "opt27b", "opt67b", "open_flamingo_mpt", "open_flamingo_redpajama", "open_flamingo_redpajama_instruct"]:
            questions = [re.sub("^<grounding> ", "", q) for q in questions]
            output = [re.split(r"[\n.]", o)[0] for o in output]
            output = [reduce(lambda txt, p: txt.partition(p)[0], partitions, text) for text in output]
            output = [o.strip() for o in output]

        for i, data in enumerate(batch):
            data["raw_prediction"] = output[i]

        if args.dataset_name in ["aokvqa", "visual7w"] and args.task_type == "multiple_choice":
            for i, data in enumerate(batch):
                data["prediction"] = data["raw_prediction"]
        else:
            cleaned_results = postprocess_vqa_answers(
                args.dataset_name, output, questions
            )  # answer batch processing for blip2
            for i, data in enumerate(batch):
                data["prediction"] = cleaned_results[i]

    return predictions
