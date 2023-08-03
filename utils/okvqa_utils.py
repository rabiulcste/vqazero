# copied from https://github.com/mlfoundations/open_flamingo and
# modified to work with our codebase.

# Those are manual mapping that are not caught by our stemming rules or would
# would be done incorrectly by our automatic stemming rule. In details,
# the keys of the _MANUAL_MATCHES dict contains the original word and the value
# contains the transformation of the word expected by the OKVQA stemming rule.
# These manual rules were found by checking the `raw_answers` and the `answers`
# fields of the released OKVQA dataset and checking all things that were not
# properly mapped by our automatic rules. In particular some of the mapping
# are sometimes constant, e.g. christmas -> christmas which was incorrectly
# singularized by our inflection.singularize.
import re
import string
from typing import List

import inflection
import nltk
import numpy as np
import spacy
from nltk.corpus.reader import VERB

_MANUAL_MATCHES = {
    "police": "police",
    "las": "las",
    "vegas": "vegas",
    "yes": "yes",
    "jeans": "jean",
    "hell's": "hell",
    "domino's": "domino",
    "morning": "morn",
    "clothes": "cloth",
    "are": "are",
    "riding": "ride",
    "leaves": "leaf",
    "dangerous": "danger",
    "clothing": "cloth",
    "texting": "text",
    "kiting": "kite",
    "firefighters": "firefight",
    "ties": "tie",
    "married": "married",
    "teething": "teeth",
    "gloves": "glove",
    "tennis": "tennis",
    "dining": "dine",
    "directions": "direct",
    "waves": "wave",
    "christmas": "christmas",
    "drives": "drive",
    "pudding": "pud",
    "coding": "code",
    "plating": "plate",
    "quantas": "quanta",
    "hornes": "horn",
    "graves": "grave",
    "mating": "mate",
    "paned": "pane",
    "alertness": "alert",
    "sunbathing": "sunbath",
    "tenning": "ten",
    "wetness": "wet",
    "urinating": "urine",
    "sickness": "sick",
    "braves": "brave",
    "firefighting": "firefight",
    "lenses": "lens",
    "reflections": "reflect",
    "backpackers": "backpack",
    "eatting": "eat",
    "designers": "design",
    "curiousity": "curious",
    "playfulness": "play",
    "blindness": "blind",
    "hawke": "hawk",
    "tomatoe": "tomato",
    "rodeoing": "rodeo",
    "brightness": "bright",
    "circuses": "circus",
    "skateboarders": "skateboard",
    "staring": "stare",
    "electronics": "electron",
    "electicity": "elect",
    "mountainous": "mountain",
    "socializing": "social",
    "hamburgers": "hamburg",
    "caves": "cave",
    "transitions": "transit",
    "wading": "wade",
    "creame": "cream",
    "toileting": "toilet",
    "sautee": "saute",
    "buildings": "build",
    "belongings": "belong",
    "stockings": "stock",
    "walle": "wall",
    "cumulis": "cumuli",
    "travelers": "travel",
    "conducter": "conduct",
    "browsing": "brows",
    "pooping": "poop",
    "haircutting": "haircut",
    "toppings": "top",
    "hearding": "heard",
    "sunblocker": "sunblock",
    "bases": "base",
    "markings": "mark",
    "mopeds": "mope",
    "kindergartener": "kindergarten",
    "pies": "pie",
    "scrapbooking": "scrapbook",
    "couponing": "coupon",
    "meetings": "meet",
    "elevators": "elev",
    "lowes": "low",
    "men's": "men",
    "childrens": "children",
    "shelves": "shelve",
    "paintings": "paint",
    "raines": "rain",
    "paring": "pare",
    "expressions": "express",
    "routes": "rout",
    "pease": "peas",
    "vastness": "vast",
    "awning": "awn",
    "boy's": "boy",
    "drunkenness": "drunken",
    "teasing": "teas",
    "conferences": "confer",
    "ripeness": "ripe",
    "suspenders": "suspend",
    "earnings": "earn",
    "reporters": "report",
    "kid's": "kid",
    "containers": "contain",
    "corgie": "corgi",
    "porche": "porch",
    "microwaves": "microwave",
    "batter's": "batter",
    "sadness": "sad",
    "apartments": "apart",
    "oxygenize": "oxygen",
    "striping": "stripe",
    "purring": "pure",
    "professionals": "profession",
    "piping": "pipe",
    "farmer's": "farmer",
    "potatoe": "potato",
    "emirates": "emir",
    "womens": "women",
    "veteran's": "veteran",
    "wilderness": "wilder",
    "propellers": "propel",
    "alpes": "alp",
    "charioteering": "chariot",
    "swining": "swine",
    "illness": "ill",
    "crepte": "crept",
    "adhesives": "adhesive",
    "regent's": "regent",
    "decorations": "decor",
    "rabbies": "rabbi",
    "overseas": "oversea",
    "travellers": "travel",
    "casings": "case",
    "smugness": "smug",
    "doves": "dove",
    "nationals": "nation",
    "mustange": "mustang",
    "ringe": "ring",
    "gondoliere": "gondolier",
    "vacationing": "vacate",
    "reminders": "remind",
    "baldness": "bald",
    "settings": "set",
    "glaced": "glace",
    "coniferous": "conifer",
    "revelations": "revel",
    "personals": "person",
    "daughter's": "daughter",
    "badness": "bad",
    "projections": "project",
    "polarizing": "polar",
    "vandalizers": "vandal",
    "minerals": "miner",
    "protesters": "protest",
    "controllers": "control",
    "weddings": "wed",
    "sometimes": "sometime",
    "earing": "ear",
}


class OKVQAStemmer:
    """Stemmer to match OKVQA v1.1 procedure."""

    def __init__(self):
        self._wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()

    def stem(self, input_string):
        """Apply stemming."""
        word_and_pos = nltk.pos_tag(nltk.tokenize.word_tokenize(input_string))
        stemmed_words = []
        for w, p in word_and_pos:
            if w in _MANUAL_MATCHES:
                w = _MANUAL_MATCHES[w]
            elif w.endswith("ing"):
                w = self._wordnet_lemmatizer.lemmatize(w, VERB)
            elif p.startswith("NNS") or p.startswith("NNPS"):
                w = inflection.singularize(w)
            stemmed_words.append(w)
        return " ".join(stemmed_words)


stemmer = OKVQAStemmer()


def postprocess_ok_vqa_generation_flamingo(predictions) -> str:
    prediction = re.split("Question|Answer", predictions, 1)[0]
    prediction_stem = stemmer.stem(prediction)
    return prediction_stem


def postprocess_generation_openflamingo(predictions) -> str:
    prediction = re.split("Question", predictions, 1)[0]
    prediction = prediction.translate(str.maketrans("", "", string.punctuation)).lower()
    # print("prediction: ", prediction)
    prediction_stem = stemmer.stem(prediction)
    return prediction_stem


dont_change_list = ["left"]


class VQAAnswerProcessor:
    """Processor for VQA answer strings."""

    def __init__(self):
        self.lemmatizer = spacy.load("en_core_web_sm", disable=["parser", "ner"])

    def process_standard(self, answers: List[str], questions=None):
        if questions is None:
            questions = [""] * len(answers)
        answers =[self.remove_punctuation(answer) for answer in answers]
        answers = [self.fix_prefix(answer) for answer in answers]
        answers = [self.remove_articles(answer) for answer in answers]
        answers = self.remove_last_noun_from_answer_if_matches_question_batch(answers, questions)
        lemmatized_answers = self.lemmatize(answers)
        lemmatized_answers = [answer.strip() for answer in lemmatized_answers]
        lemmatized_answers = [answer.lower() for answer in lemmatized_answers]
        return lemmatized_answers

    def process_aokvqa_batch(self, predictions: List[str]):
        predictions = [prediction.lower() for prediction in predictions]
        predictions = [self.fix_prefix(prediction) for prediction in predictions]
        predictions = [self.remove_articles(prediction) for prediction in predictions]
        # predictions = self.lemmatize(predictions)
        predictions = [input_string.strip(string.punctuation) for input_string in predictions]  # remove punctuation
        predictions = [input_string.strip() for input_string in predictions] # remove leading and trailing whitespaces
        return predictions
    
    def process_aokvqa(self, prediction: str):
        prediction = prediction.lower()
        prediction = self.fix_prefix(prediction)
        prediction = self.remove_articles(prediction)
        prediction = self.lemmatize([prediction])[0]
        prediction = prediction.strip(string.punctuation)  # remove punctuation
        prediction = prediction.strip()  # remove leading and trailing whitespaces
        return prediction

    def process_gqa(self, answers: List[str], questions=None):
        if questions is None:
            questions = [""] * len(answers)
        answers = [self.remove_punctuation(answer) for answer in answers]
        answers = [self.fix_prefix(answer) for answer in answers]
        answers = [self.remove_articles(answer) for answer in answers]
        answers = self.remove_last_noun_from_answer_if_matches_question_batch(answers, questions)
        answers = [answer.strip() for answer in answers]
        answers = [answer.lower() for answer in answers]
        return answers

    @staticmethod
    def remove_punctuation(word):
        # Define a regular expression pattern that matches punctuation marks at the beginning or end of the word
        pattern = r"^[\W_]+|[\W_]+$"

        # Use the sub() method of the re module to remove the punctuation marks
        return re.sub(pattern, "", word)

    def lemmatize(self, answers):
        lemmatized_answers = []
        dont_change_set = set(dont_change_list)
        for doc in self.lemmatizer.pipe(answers, batch_size=64, n_process=-1):
            words = []
            for token in doc:
                if token.pos_ in ["NOUN", "VERB"] and token.text not in dont_change_set:
                    words.append(token.lemma_)
                else:
                    words.append(token.text)
            lemmatized_answer = " ".join(words)
            lemmatized_answers.append(lemmatized_answer)
        return lemmatized_answers

    def remove_last_noun_from_answer_if_matches_question_batch(self, answers, questions):
        """Remove answer suffix if it is a noun found in the corresponding question."""
        question_nouns = []
        for doc in self.lemmatizer.pipe(questions, batch_size=1000, n_process=-1):
            nouns = []
            for token in doc:
                if token.pos_ == "NOUN":
                    nouns.append(token.text)
            question_nouns.append(nouns)

        processed_answers = []
        for answer, nouns in zip(answers, question_nouns):
            answer_words = answer.split()
            if len(answer_words) < 2:
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


def postprocess_batch_vqa_generation_blip2(dataset_name: str, predictions: List[str], questions=None) -> str:
    if dataset_name == "aokvqa":
        predictions = answer_processor.process_aokvqa_batch(predictions)
    elif dataset_name == "gqa":
        predictions = answer_processor.process_gqa(predictions, questions)
    else:
        predictions = answer_processor.process_standard(predictions, questions)

    return predictions


def extract_answer_from_exemplars(qa_pairs):
    answers = []
    for qa in qa_pairs:
        match = re.search(r'[Aa]nswer: "(.*?)"', qa)
        if match:
            answer = match.group(1)
            answers.append(answer)
    return answers


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
