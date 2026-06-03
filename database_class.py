import json
import joblib
import numpy as np
from pprint import pprint

from vectorizers.Empath4D import Empath4DVectorMaker
from vectorizers.NRC_EIL import NRC_EIL_VectorMaker
from vectorizers.SentanceBERT import SBERTVectorMaker
from STEM_classification.cosine_classifier import CosineClassifier
from helper_functions import combine_using_bilinear_pool

def get_stem_topic_set():
    with open("STEM_classification/STEM_topics.txt", "r", encoding="utf-8") as file:
        stem_set = {
            line.strip() for line in file if line.strip()
        }

    return stem_set

class Database():
    def __init__(self, data_file = "test_files/sampled_output_3_4.jsonl"):
        self.TOPIC_VECTOR_MAKER = Empath4DVectorMaker()
        self.EMOTION_VECTOR_MAKER = NRC_EIL_VectorMaker()
        self.CLASSIFIER_VECTOR_MAKER = SBERTVectorMaker()
        self.CLASSIFIER_MODEL = joblib.load("STEM_classification/best_model_GaussianNB.joblib")
        self.COSINE_CLASSIFIER = CosineClassifier()
        self.STEM_TOPIC_SET = get_stem_topic_set()
        json_lines = self.read_in_jsonl_to_list(data_file)
        self.database = self.create_dataset(json_lines)

    def process_book(self, book_json_object):
        isbn = book_json_object.get("ISBN")
        description = book_json_object.get("description")

        topic_vec = self.TOPIC_VECTOR_MAKER.getEmapthVector(description)
        emotion_vec = self.EMOTION_VECTOR_MAKER.getEmotionVector(description)

        # STEM CLASSIFICATION 
        is_STEM = False
        loc_subjects = book_json_object.get("LoC_subjects", [])
        google_categories = book_json_object.get("Google_categories", [])

        has_loc_match = any(subject in self.STEM_TOPIC_SET for subject in loc_subjects)
        has_google_match = any(
            category in self.STEM_TOPIC_SET for category in google_categories
        )

        if has_loc_match or has_google_match:
            is_STEM = True
        else:
            vector_for_classification = self.CLASSIFIER_VECTOR_MAKER.get_vector(description)
            formatted_vector_for_classification = np.array(vector_for_classification).reshape(1, -1)

            prediction = self.CLASSIFIER_MODEL.predict(formatted_vector_for_classification)
            predicted_label = prediction[0]
            if predicted_label == 1:
                is_STEM = True
            else:
                if self.COSINE_CLASSIFIER.is_stem(topic_vec):
                    is_STEM = True

        book = {
            "isbn" : isbn,
            "topic_vec" : topic_vec,
            "emotion_vec" : emotion_vec,
            "is_stem" : is_STEM
        }
        return book

    def create_dataset(self, jsonl_list):
        database = []
        for book in jsonl_list:
            database.append(self.process_book(book))
        
        database_map = {
            item['isbn']: {
                'topic_vec': item['topic_vec'],
                'emotion_vec': item['emotion_vec'],
                'is_stem': item['is_stem']
            } 
            for item in database
        }
        return database_map

    def read_in_jsonl_to_list(self, path):
        results = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                results.append(json.loads(line))
        return results

    def get_topic_vec_for_book(self, isbn):
        return self.database[isbn]['topic_vec']
    
    def get_emotion_vec_for_book(self, isbn):
        return self.database[isbn]['emotion_vec']
    
    def is_book_STEM(self, isbn):
        return self.database[isbn]['is_stem']
    
    def add_book(self, book_json):
        transformed_book_json = self.process_book(book_json)

        self.database[transformed_book_json["isbn"]] = {
            "topic_vec": transformed_book_json["topic_vec"],
            "emotion_vec": transformed_book_json["emotion_vec"],
            "is_stem": transformed_book_json["is_stem"]
        }
        # this currently doesn't add anything to long term storage
        # so only there while program runs 

database = Database()
print(type(database.is_book_STEM('0970331010')))