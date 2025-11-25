import re
import json
import torch
from itertools import chain
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Download required NLTK data
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet', quiet=True)
try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger', quiet=True)
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)

class TextPreprocessor:
    def __init__(self, keywords_file: str = None, debug: bool = False):
        """
        :param keywords_file: JSON file to load/save new accessibility keywords
        """
        self.debug = debug

        # Default keywords file path
        if keywords_file is None:
            keywords_file = "src/utils/accessibility_keywords.json"
        
        # Load initial keywords
        try:
            with open(keywords_file, "r") as f:
                self.keywords = json.load(f)
        except FileNotFoundError:
            # Fallback to empty list if file not found
            self.keywords = []

        # SBERT model for semantic similarity
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.keyword_embeddings = self.model.encode(self.keywords, convert_to_tensor=True)

        # file to persist keywords
        self.keywords_file = keywords_file
        
        # Initialize lemmatizer and articles list
        self.lemmatizer = WordNetLemmatizer()
        self.articles = {'a', 'an', 'the'}
        self.stop_words = set(stopwords.words('english'))
        # POS tags for nouns and verbs
        self.allowed_pos = {'NN', 'NNS', 'NNP', 'NNPS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ'}

    # Text cleaning
    def clean_text(self, text: str, is_sentiment: bool = False, remove_articles: bool = False, lemmatize: bool = False) -> str:
        text = text.lower()
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'[-/]', ' ', text)
        if is_sentiment:
            text = re.sub(r'[^a-zA-Z0-9\s!?.,]', '', text)
        else:
            text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Remove articles if requested
        if remove_articles:
            words = text.split()
            words = [word for word in words if word not in self.articles]
            text = ' '.join(words)
        
        # Lemmatize if requested
        if lemmatize:
            words = text.split()
            words = [self.lemmatizer.lemmatize(word) for word in words]
            text = ' '.join(words)
        
        return text

    # Filter tokens to only keep nouns and verbs
    def filter_nouns_verbs(self, text: str) -> str:
        tokens = text.split()
        pos_tags = nltk.pos_tag(tokens)
        filtered = [word for word, pos in pos_tags if pos in self.allowed_pos]
        return ' '.join(filtered)
    
    # Generate n-grams
    def generate_ngrams(self, tokens, n=3):
        ngrams = []
        for size in range(1, n+1):
            for i in range(len(tokens) - size + 1):
                ngram = " ".join(tokens[i:i+size])
                ngrams.append(ngram)
        return ngrams

    # Extract accessibility keywords from text
    def extract_accessibility_keywords(
        self,
        text: str,
        threshold: float = 0.75,
        expand_keywords: bool = True
    ) -> list:
        cleaned_text = self.clean_text(text, remove_articles=True, lemmatize=True)
        tokenized_text = cleaned_text.split()
        extracted_keywords = set()

        # Generate n-grams and embeddings
        ngrams = self.generate_ngrams(tokenized_text, n=3)
        ngram_embeddings = self.model.encode(ngrams, convert_to_tensor=True)
        cosine_scores = util.cos_sim(ngram_embeddings, self.keyword_embeddings)

        for i, ngram in enumerate(ngrams):
            # Check similarity with existing keywords
            for j, keyword in enumerate(self.keywords):
                score = float(cosine_scores[i][j])
                if score >= threshold:
                    extracted_keywords.add(keyword)
                    if self.debug:
                        print(f"N-gram '{ngram}' is similar to keyword '{keyword}' ({score:.2f})")

            # Expand keyword list if new phrase is very similar (>= 0.8)
            if expand_keywords:
                max_score = float(torch.max(cosine_scores[i]).item())
                if max_score >= 0.8:
                    sim_idx = int(torch.argmax(cosine_scores[i]).item())
                    # Get the most similar keyword before adding new one
                    most_similar_keyword = self.keywords[sim_idx] if sim_idx < len(self.keywords) else self.keywords[0]
                    new_keyword = ngrams[i]
                    # Filter to only keep nouns and verbs
                    filtered_keyword = self.filter_nouns_verbs(new_keyword)
                    if filtered_keyword and filtered_keyword not in self.keywords:
                        self.keywords.append(filtered_keyword)
                        # Recompute embeddings
                        self.keyword_embeddings = self.model.encode(self.keywords, convert_to_tensor=True)
                        # Recompute cosine scores for remaining ngrams
                        cosine_scores = util.cos_sim(ngram_embeddings, self.keyword_embeddings)
                        if self.debug:
                            print(f"Added new keyword: '{filtered_keyword}' (similar to '{most_similar_keyword}', score={max_score:.2f})")

        if self.debug:
            print(f"Extracted Keywords: {list(extracted_keywords)}")

        # Save keywords if file provided
        if self.keywords_file:
            with open(self.keywords_file, "w") as f:
                json.dump(self.keywords, f)

        return list(extracted_keywords)
