import json
import joblib
import numpy as np
from pprint import pprint
import numpy as np
from sklearn.neighbors import NearestNeighbors

from vectorizers.Empath4D import Empath4DVectorMaker
from vectorizers.NRC_EIL import NRC_EIL_VectorMaker
from vectorizers.SentanceBERT import SBERTVectorMaker
from STEM_classification.cosine_classifier import CosineClassifier
from helper_functions import combine_using_bilinear_pool, average_vectors

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
        title = book_json_object.get("Book-Title", "")
        author = book_json_object.get("Book-Author", "")

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
            "title_and_author": f"{title} by {author}",
            "topic_vec" : topic_vec,
            "emotion_vec" : emotion_vec,
            "is_stem" : is_STEM
        }
        return book

def read_in_jsonl_to_list(path):
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            results.append(json.loads(line))
    return results

class Database():
    def __init__(self, data_file = "test_files/sampled_output_3_4.jsonl"):
        json_lines = read_in_jsonl_to_list(data_file)
        self.database = self.create_dataset(json_lines)
        self.create_look_up()

    def create_dataset(self, jsonl_list):
        database = []
        for book in jsonl_list:
            database.append(process_book(book))
        
        database_map = {
            item['isbn']: {
                'topic_vec': item['topic_vec'],
                'emotion_vec': item['emotion_vec'],
                'is_stem': item['is_stem'],
                'title_and_author': item['title_and_author'],
            } 
            for item in database
        }
        return database_map

    def create_look_up(self):
        self.book_output = []
        self.metadata_registry = []
        vector_list = []
        for isbn, info in self.database.items():
            combined_vec = combine_using_bilinear_pool(info['emotion_vec'], info['topic_vec'])
            vector_list.append(combined_vec)
            self.book_output.append(info['title_and_author'])
            self.metadata_registry.append({'is_stem': info['is_stem']})
        
        X = np.array(vector_list)
        self.nn_model = NearestNeighbors(n_neighbors=len(self.book_output), metric='cosine')
        self.nn_model.fit(X)

    def get_recommendation(self, user):
        query_vector = user.user_profile_vec.reshape(1, -1)
        distances, indices = self.nn_model.kneighbors(query_vector)

        for rank, (idx, dist) in enumerate(zip(indices[0][1:], distances[0][1:]), start=1):
            match_book = self.book_output[idx]
            match_meta = self.metadata_registry[idx]
            
            # Calculate similarity percentage from cosine distance
            similarity = (1 - dist) * 100
            
            print(f"Rank {rank}: {match_book}")
            print(f" -> Similarity Score: {similarity:.2f}%")
            print(f" -> Is STEM Book: {match_meta['is_stem']}\n")

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
        self.create_look_up() # need to redo look up
        # this currently doesn't add anything to long term storage
        # so only there while program runs 




class User():
    def __init__(self, data_file = "test_files/sampled_output_2_4.jsonl"):
        json_lines = read_in_jsonl_to_list(data_file)
        self.isbn_list = []
        book_vecs = []
        for line in json_lines:
            book = process_book(line)
            self.isbn_list.append(book['isbn'])
            combined_book_vec = combine_using_bilinear_pool(book['emotion_vec'], book['topic_vec'])
            book_vecs.append(combined_book_vec)
        self.user_profile_vec = average_vectors(book_vecs)
            
    
database = Database()
user = User()
database.get_recommendation(user)