import json
import joblib
import numpy as np

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

TOPIC_VECTOR_MAKER = Empath4DVectorMaker()
EMOTION_VECTOR_MAKER = NRC_EIL_VectorMaker()
CLASSIFIER_VECTOR_MAKER = SBERTVectorMaker()
CLASSIFIER_MODEL = joblib.load("STEM_classification/best_model_GaussianNB.joblib")
COSINE_CLASSIFIER = CosineClassifier()
STEM_TOPIC_SET = get_stem_topic_set()

def process_book(book_json_object):
    isbn = book_json_object.get("ISBN")
    description = book_json_object.get("description")

    topic_vec = TOPIC_VECTOR_MAKER.getEmapthVector(description)
    emotion_vec = EMOTION_VECTOR_MAKER.getEmotionVector(description)

    # STEM CLASSIFICATION 
    is_STEM = False
    loc_subjects = book_json_object.get("LoC_subjects", [])
    google_categories = book_json_object.get("Google_categories", [])

    has_loc_match = any(subject in STEM_TOPIC_SET for subject in loc_subjects)
    has_google_match = any(
        category in STEM_TOPIC_SET for category in google_categories
    )

    if has_loc_match or has_google_match:
        is_STEM = True
    else:
        vector_for_classification = CLASSIFIER_VECTOR_MAKER.get_vector(description)
        formatted_vector_for_classification = np.array(vector_for_classification).reshape(1, -1)

        prediction = CLASSIFIER_MODEL.predict(formatted_vector_for_classification)
        predicted_label = prediction[0]
        if predicted_label == 1:
            is_STEM = True
        else:
            if COSINE_CLASSIFIER.is_stem(topic_vec):
                is_STEM = True

    book = {
        "isbn" : isbn,
        "topic_vec" : topic_vec,
        "emotion_vec" : emotion_vec,
        "is_stem" : is_STEM
    }
    return book

def create_dataset(jsonl_list):
    database = []
    for book in jsonl_list:
        database.append(process_book(book))
    return database


def read_in_jsonl_to_list(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results

# test_1 = read_in_jsonl_to_list("test_files/sampled_output_3_4.jsonl")
# create_dataset(test_1)