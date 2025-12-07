from src.preprocessing import TextPreprocessor
from src.preprocessing import FeatureExtractor
from src.preprocessing import KeywordManager
from src.preprocessing import KeywordExtractor
from src.model import SentimentModel

class AccessibilityPipeline:
    def __init__(self, preprocessor: TextPreprocessor, feature_extractor: FeatureExtractor, keyword_manager: KeywordManager, keyword_extractor: KeywordExtractor, sentiment_model: SentimentModel):
        self.preprocessor = preprocessor
        self.feature_extractor = feature_extractor
        self.keyword_manager = keyword_manager
        self.keyword_extractor = keyword_extractor
        self.sentiment_model = sentiment_model

    def analyze_review(self, text: str, save_keywords: bool = False) -> dict:
        # Preprocess the text for analysis
        cleaned_text = self.preprocessor.lowercase(text)
        cleaned_text = self.preprocessor.remove_url(cleaned_text)
        cleaned_text = self.preprocessor.remove_non_alpha(cleaned_text)
        
        # Tokenize and remove stopwords and articles
        tokens = cleaned_text.split()
        tokens = self.preprocessor.remove_stopwords(tokens)
        tokens = self.preprocessor.remove_articles(tokens)

        # Lemmatize tokens and filter by allowed POS tags
        lemmatized_tokens = self.preprocessor.lemmatize(tokens)
        filtered_tokens = self.preprocessor.filter_pos(lemmatized_tokens)

        # Generate n-grams
        ngrams = self.feature_extractor.generate_ngrams(filtered_tokens)

        # Check if review is about accessibility
        if self.keyword_extractor.is_relevant_review(ngrams):
            extracted_keywords = self.keyword_extractor.extract_keywords(ngrams) # Extract keywords
            self.keyword_manager.add_keywords(extracted_keywords) # add keywords to the keywords list
            sentiment_result = self.sentiment_model.analyze_sentiment(cleaned_text) # Analyze sentiment only for relevant reviews
        else:
            exit() # Skip non-relevant reviews

        # Return unified result
        return {
            "keywords": extracted_keywords,
            "sentiment": sentiment_result["sentiment"],
            "confidence": sentiment_result["confidence"]
        }
