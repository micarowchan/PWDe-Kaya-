from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class SentimentModel:
    def __init__(self, model=None, tokenizer=None, model_path="finetuned-accessibility-bert/final_model", device=None, sentiment_map=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.sentiment_map = sentiment_map or {0: "Negative", 1: "Mixed", 2: "Positive"}

        if model and tokenizer:
            self.model = model
            self.tokenizer = tokenizer
        elif model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            raise ValueError("Either model + tokenizer or model_path must be provided.")

        self.model.to(self.device)
        self.model.eval()

    # Analyze sentiment of a given text
    def analyze_sentiment(self, text: str) -> dict:
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Perform inference
        with torch.no_grad():
            outputs = self.model(**inputs)
            probability = F.softmax(outputs.logits, dim=-1)[0]

        # Get predicted class and confidence
        predicted_class = torch.argmax(probability).item()
        confidence = probability[predicted_class].item()
        sentiment = self.sentiment_map.get(predicted_class, "Unknown")

        return {"sentiment": sentiment, "confidence": round(confidence * 100, 1)}
