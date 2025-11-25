from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from src.preprocessing.text_preprocessor import TextPreprocessor

class SentimentModel:
    def __init__(self, model_path=None, device=None):
        """
        :param model_path: path to the fine-tuned model folder
        """
        self.model_path = model_path or "nlptown/bert-base-multilingual-uncased-sentiment"
        
        # If using local fine-tuned model, point to final_model subdirectory
        if model_path and "finetuned-accessibility-bert" in model_path:
            if not model_path.endswith("final_model"):
                self.model_path = model_path + "/final_model"
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_path)

        # Set device
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def analyze_sentiment(self, text: str) -> dict:
        # --- 1. Minimal cleaning for sentiment (preserve context) ---
        # Only remove URLs and normalize whitespace, keep punctuation and structure
        import re
        cleaned_text = text.lower()
        cleaned_text = re.sub(r'http\S+|www\S+|https\S+', '', cleaned_text)
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()

        # --- 2. Tokenize ---
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # --- 3. Forward pass ---
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]

        # --- 4. Predicted label ---
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        # --- 5. Map to human-readable sentiment ---
        sentiment_map = {0: "Negative", 1: "Mixed", 2: "Positive"}
        sentiment = sentiment_map.get(pred_class, "Unknown")

        return {
            "sentiment": sentiment,
            "confidence": round(confidence * 100, 1),  # Convert to percentage
        }
