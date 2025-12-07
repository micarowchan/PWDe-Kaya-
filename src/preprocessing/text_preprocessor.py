import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# Ensure NLTK resources are downloaded
for resource in ['wordnet', 'omw-1.4', 'stopwords', 'averaged_perceptron_tagger']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class TextPreprocessor:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.articles = {'a', 'an', 'the'}
        self.allowed_pos = {
            'NN','NNS','NNP','NNPS',      # nouns
            'VB','VBD','VBG','VBN','VBP','VBZ',  # verbs
            'JJ','JJR','JJS',              # adjectives
            'RB','RBR','RBS'               # adverbs
        }

    # lowercase text
    def lowercase(self, text: str) -> str:
        return text.lower()

    # remove URLs
    def remove_url(self, text: str) -> str:
        return re.sub(r'http\S+|www\S+|https\S+', '', text)

    # remove non-alphabetic characters
    def remove_non_alpha(self, text: str) -> str:
        # Remove non-alphabetic characters, replace hyphens/slashes with space
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = re.sub(r'[-/]', ' ', text)
        return text
    
    # lemmatize into tokens
    def lemmatize(self, tokens: list) -> list:
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in tokens]
        return lemmatized_tokens
    
    # remove stopwords
    def remove_stopwords(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.stop_words]

    # remove articles
    def remove_articles(self, tokens: list) -> list:
        return [t for t in tokens if t not in self.articles]

    # select only words according to allowed POS tags
    def filter_pos(self, tokens: list, pos_tags: list = None) -> list:
        if pos_tags is None:
            pos_tags = nltk.pos_tag(tokens)
        return [token for token, tag in pos_tags if tag in self.allowed_pos]
    