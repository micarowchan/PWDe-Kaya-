from src.preprocessing.text_preprocessor import TextPreprocessor
from src.model.sentiment_model import SentimentModel

text_preprocessor = TextPreprocessor(debug=False)
sentiment_model = SentimentModel(model_path="finetuned-accessibility-bert")

review_samples = [
    "It's okay. They're PWD-friendly. They do have a person or staff at the lobby but not as approachable as they should be. Went here a couple of times to view exhibits and when I try to ask, I find they are not that friendly but had a nice visit anyway.",
    "They have a few parking slots that are shared with the nearby establishments. Items are sorted neatly according to purpose, so it's easy to find exactly where to get what you need. They accept credit card payment. They have a self service GoTyme kiosk, as well as a counter dedicated for GoTyme deposit and withdrawal.",
    "The Food was good."
]

for review in review_samples:
    keywords = text_preprocessor.extract_accessibility_keywords(review)
    if keywords:  # If keywords were found, it's accessible
        sentiment = sentiment_model.analyze_sentiment(review)
        print(f"Review: {review}\n")
        print(f"Keywords: {keywords}")
        print(f"Sentiment: {sentiment['sentiment']} ({sentiment['confidence']}% confident)\n")