import copy
import datetime
import json
import os

from utils.config import VQA_DATASET_DIR
from utils.globals import DATASET_CONFIG


class VQA:
    def __init__(self, dataset_name: str, annotation_file: str = None, question_file: str = None):
        dataset_dir = os.path.join(VQA_DATASET_DIR, "datasets")
        self.dataset_name = dataset_name
        self.data_dir = os.path.join(dataset_dir, dataset_name)
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}

        if annotation_file is None and question_file is None:
            question_file = DATASET_CONFIG[dataset_name]["val"]["question_file"]
            annotation_file = DATASET_CONFIG[dataset_name]["val"]["annotation_file"]
            with open(os.path.join(self.data_dir, annotation_file)) as f:
                dataset = json.load(f)
            with open(os.path.join(self.data_dir, question_file)) as f:
                questions = json.load(f)
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann

        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques

        print("index created!")

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print("%s: %s" % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                quesTypes (str array)   : get question ids for given question types
                ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                anns = sum([self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA], [])
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann["question_type"] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann["answer_type"] in ansTypes]
        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
        Get image ids that satisfy given filter conditions. default skips that filter
        :param quesIds   (int array)   : get image ids for given question ids
               quesTypes (str array)   : get image ids for given question types
               ansTypes  (str array)   : get image ids for given answer types
        :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(quesIds) == 0:
                anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa], [])
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann["question_type"] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann["answer_type"] in ansTypes]
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            print("Question: %s" % (self.qqa[quesId]["question"]))
            for ans in ann["answers"]:
                print("Answer %d: %s" % (ans["answer_id"], ans["answer"]))

    def loadRes(self, resFile=None, quesFile=None):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        # res.questions = json.load(open(quesFile))
        if quesFile is None:
            quesFile = DATASET_CONFIG[self.dataset_name]["val"]["question_file"]
            quesFile = os.path.join(self.data_dir, quesFile)

        res = VQA(self.dataset_name, question_file=quesFile)
        with open(quesFile) as f:
            res.questions = json.load(f)
        res.dataset["info"] = copy.deepcopy(self.questions["info"])
        res.dataset["task_type"] = copy.deepcopy(self.questions["task_type"])
        res.dataset["data_type"] = copy.deepcopy(self.questions["data_type"])
        res.dataset["data_subtype"] = copy.deepcopy(self.questions["data_subtype"])
        res.dataset["license"] = copy.deepcopy(self.questions["license"])

        print("Loading and preparing results...     ")
        time_t = datetime.datetime.utcnow()
        # anns = json.load(open(resFile))
        if resFile is None:
            raise ValueError("resFile must be specified")
        print(resFile)
        with open(resFile) as f:
            anns = json.load(f)
        assert type(anns) == list, "results is not an array of objects"
        annsQuesIds = [ann["question_id"] for ann in anns]
        assert set(annsQuesIds) == set(
            self.getQuesIds()
        ), "Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file."
        for ann in anns:
            quesId = ann["question_id"]
            if res.dataset["task_type"] == "Multiple Choice":
                assert (
                    ann["answer"] in self.qqa[quesId]["multiple_choices"]
                ), "predicted answer is not one of the multiple choices"
            qaAnn = self.qa[quesId]
            ann["image_id"] = qaAnn["image_id"]
            ann["question_type"] = qaAnn["question_type"]
            ann["answer_type"] = qaAnn["answer_type"]
        print("DONE (t=%0.2fs)" % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res
