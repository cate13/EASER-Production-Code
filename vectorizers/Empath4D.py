from empath import Empath

class Empath4DVectorMaker:
    def __init__(self):
        self.lexicon = Empath()

        self.lexicon.create_category("engineering", ["engineering", "chemical_engineering", "electrical_engineering", "mechanical_engineering", "civil_engineering"], model="reddit", size=300)

        self.lexicon.create_category("mathematics", ["math", "mathematics", "arithmetic", "algebra", "geometry", "calculus", "statistics", "trigonometry" ], model="reddit", size=300)

    def getEmapthVector(self, text, categories=["science", "technology", "engineering", "mathematics"]):
        return self.lexicon.analyze(text, categories=categories, normalize=True)
    

