import ast
import argparse
import os
import numpy as np
from collections import defaultdict


class Cleaner(ast.NodeVisitor):
    def __init__(self, text):
        try:
            self.root = ast.parse(text)
        except:
            self.root = ast.parse("")

    def get_clean_code(self):
        self.delete_docstrings()
        return ast.unparse(self.root)

    def delete_docstrings(self):
        for node in ast.walk(self.root):
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef)):
                if ast.get_docstring(node) is not None:
                    node.body = node.body[1:]


class PlagiarismDetectionModel():
    def __init__(self):
        self.idf = None
        self.vocabulary = dict()
        self.all_words = None

    def fit(self, files_path, plagiat1_path, plagiat2_path):
        initial_texts = np.array(
            [Cleaner(open(os.path.join(files_path, filename)).read()).get_clean_code() for filename in
             os.listdir(files_path)])

        plagiat1_texts = np.array(
            [Cleaner(open(os.path.join(plagiat1_path, filename)).read()).get_clean_code() for filename in
             os.listdir(plagiat1_path)])

        plagiat2_texts = np.array(
            [Cleaner(open(os.path.join(plagiat2_path, filename)).read()).get_clean_code() for filename in
             os.listdir(plagiat2_path)])
        all_files = np.concatenate((initial_texts, plagiat1_texts, plagiat2_texts))

        df, self.vocabulary = self.getDF(all_files)
        self.all_words = list(self.vocabulary.keys())
        self.idf = np.empty(len(self.vocabulary.keys()))

        for index, word in enumerate(self.all_words):
            self.idf[index] = np.log(all_files.size / df[word]) + 1

    @staticmethod
    def getDF(documents):
        df = dict()
        frequency = dict()
        general_count = 0
        for text in documents:
            is_used = defaultdict(bool)
            for word in text.split():
                general_count += 1
                frequency[word] = frequency.get(word, 0) + 1
                if not is_used[word]:
                    df[word] = df.get(word, 0) + 1
                    is_used[word] = True
        return df, frequency

    @staticmethod
    def getTF(text: str):
        frequency = dict()
        general_count = 0
        for word in text.split():
            general_count += 1
            frequency[word] = frequency.get(word, 0) + 1
        return frequency, general_count

    def getVector(self, text1: str, text2: str):
        first_dict, first_count = self.getTF(text1)
        second_dict, second_count = self.getTF(text2)
        first_vector = np.empty(len(self.all_words))
        second_vector = np.empty(len(self.all_words))
        for index in range(len(self.all_words)):
            first_vector[index] = first_dict.get(self.all_words[index], 0) * self.idf[index]
            second_vector[index] = second_dict.get(self.all_words[index], 0) * self.idf[index]
        return first_vector, second_vector

    def predict(self, pair):
        code1 = Cleaner(open(pair[0]).read()).get_clean_code()
        code2 = Cleaner(open(pair[1]).read()).get_clean_code()
        vector1, vector2 = self.getVector(code1, code2)
        score = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        return str(score)


detector = PlagiarismDetectionModel()
detector.fit("./data/files/", "./data/plagiat1/", "./data/plagiat2/")

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
parser.add_argument('output', type=str)
args = parser.parse_args()

output_file = open(args.output, "w+")
with open(args.input) as input_file:
    for line in input_file.readlines():
        output_file.write(detector.predict(line.split()) + '\n', )
output_file.close()
