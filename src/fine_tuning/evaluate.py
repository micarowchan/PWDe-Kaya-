# src/fine_tuning/evaluate.py
import torch
import torch.nn.functional as F
import pandas as pd
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Load fine-tuned model and tokenizer
def load_model(model_dir: str):
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    return tokenizer, model

# Load dataset
def load_data(file_path: str):
    file_data = pd.read_csv(file_path)
    return file_data

# Predict labels for the test dataset
def predict(model, tokenizer, texts):
    predicted_labels = []
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
            probability = F.softmax(outputs.logits, dim=-1)
            prediction = torch.argmax(probability, dim=-1).item()
            predicted_labels.append(prediction)
    return predicted_labels

# Evaluate model performance
def evaluate(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_recall_fscore_support(y_true, y_pred, average='macro')[0]
    recall = precision_recall_fscore_support(y_true, y_pred, average='macro')[1]
    f1 = precision_recall_fscore_support(y_true, y_pred, average='macro')[2]
    conf_matrix = confusion_matrix(y_true, y_pred)
    return accuracy, precision, recall, f1, conf_matrix

# Export evaluation results to JSON
def export_results_to_json(model_name, accuracy, precision, recall, f1, output_file="output/evaluation_results.json"):
    results = {
        "model_name": model_name,
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "evaluation_date": datetime.now().isoformat()
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    return results

def main():
    model_dir = "./finetuned-accessibility-bert/final_model"
    test_csv = "data/test_reviews.csv"

    tokenizer = load_model(model_dir)[0]
    model = load_model(model_dir)[1]

    test_df = load_data(test_csv)

    y_true = test_df["label"].tolist()
    predicted_labels = predict(model, tokenizer, test_df['review_text'])

    accuracy, precision, recall, f1, conf_mat = evaluate(y_true, predicted_labels)
    label_map_inv = {0: 'Negative', 1: 'Mixed', 2: 'Positive'}

    # Display evaluation results
    print("\nEvaluation Results:")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1-score: {f1:.2f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(
        conf_mat,
        index=[label_map_inv[i] for i in range(3)],
        columns=[label_map_inv[i] for i in range(3)]
    ))

    # Export to JSON for backend
    model_name = "RoBERTa-finetuned"
    export_results_to_json(model_name, accuracy, precision, recall, f1)


if __name__ == "__main__":
    main()
