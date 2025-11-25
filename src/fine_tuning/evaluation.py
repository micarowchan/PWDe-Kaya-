# src/fine_tuning/evaluate.py
import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

def main():
    # Load test dataset
    test_csv = "data/test_reviews.csv"
    test_df = pd.read_csv(test_csv)

    # Load fine-tuned model
    model_dir = "./finetuned-accessibility-bert/final_model"
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    # Map labels to numbers
    label_map = {"Positive": 0, "Mixed": 1, "Negative": 2}
    y_true = test_df['label'].map(label_map).tolist()

    # Predict
    pred_labels = []
    for text in test_df['review_text']:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            pred_labels.append(pred)

    # Metrics
    accuracy = accuracy_score(y_true, pred_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_labels, average='macro')
    cm = confusion_matrix(y_true, pred_labels)

    # Map numbers back to labels for confusion matrix
    label_map_inv = {v: k for k, v in label_map.items()}

    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=[label_map_inv[i] for i in range(3)],
                             columns=[label_map_inv[i] for i in range(3)]))

if __name__ == "__main__":
    main()
