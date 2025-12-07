import json

class KeywordManager:
    def __init__(self, keywords_file: str):
        self.keywords_file = keywords_file
        self.keywords = self._load_keywords()

    def _load_keywords(self):
        try:
            with open(self.keywords_file, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []

    def save_keywords(self):
        with open(self.keywords_file, "w") as f:
            json.dump(self.keywords, f)

    def add_keywords(self, new_keywords: list, save: bool = True):
        self.keywords.extend(new_keywords)
        self.keywords = list(set(self.keywords)) # Remove duplicates   
        if save:
            self.save_keywords()
