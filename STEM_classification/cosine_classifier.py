import os
import sys
import json

# Get the path of the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Add the parent directory to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from helper_functions import average_vectors

class CosineClassifier:
    def __init__(self):
        all_stem_vecs = []
        with open("STEM_classification/sample_stem_with_empath_4d.jsonl", 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                data = json.loads(line)
                all_stem_vecs.append(data.get('empath4D'))
        self.average_STEM_vec = average_vectors(all_stem_vecs)
