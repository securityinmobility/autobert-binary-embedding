#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import numpy as np
from scipy.spatial import KDTree


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def find_first(haystack, needles):
    for needle in needles:
        if needle in haystack:
            return needle
    return None

class DocumentIndexer:
    def __init__(self, jsonl_file_path):
        self.jsonl_file_path = jsonl_file_path
        self.kd_tree = None
        self.load_documents()

    def load_documents(self):
        try:
            self.documents = []
            vectors = []
            with open(self.jsonl_file_path, 'r') as file:
                for line in file:
                    doc_data = json.loads(line)
                    self.documents.append(doc_data)
                    vectors.append(doc_data['vector'])
            self.build_kd_tree(vectors)
        except (FileNotFoundError, json.JSONDecodeError):
            # If the file doesn't exist or is not valid JSON, initialize with an empty list
            self.documents = []
            self.kd_tree = None

    def build_kd_tree(self, vectors):
        if vectors:
            self.kd_tree = KDTree(vectors)

    def save_document(self, document):
        with open(self.jsonl_file_path, 'a') as file:
            json.dump(document, file)
            file.write('\n')

    def add_document(self, vector, metadata):
        new_document = {'vector': vector.tolist(), 'metadata': metadata}
        self.documents.append(new_document)
        if self.kd_tree is not None:
            vectors = [doc['vector'] for doc in self.documents]
            self.build_kd_tree(vectors)
        self.save_document(new_document)

    def query_nearest_documents(self, query_vector, k=5):
        if self.kd_tree:
            dists, indices = self.kd_tree.query(query_vector, k)
            nearest_documents = [self.documents[i] for i in indices]
            return nearest_documents, dists
        else:
            return [], []
