# src/fine_tuning/fine_tune.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np
import os

def load_data(train_csv: str, test_csv: str):
    dataset = load_dataset("csv", data_files={"train": train_csv, "test": test_csv})
    return dataset

def map_labels(dataset):
    # Convert labels to integers 
    label_map = {"Positive": 2, "Mixed": 1, "Negative": 0}
    
    def map_example(example):
        label = example["label"]
        # If label is string, map it to int
        if isinstance(label, str):
            example["label"] = label_map[label]
        else:
            # If already an integer, put it in correct range
            example["label"] = int(label)
        return example
    
    return dataset.map(map_example)

# Trainer with class weights for imbalanced data
class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss

def main():
    # Dataset paths
    train_csv = "data/train_reviews.csv"
    test_csv = "data/test_reviews.csv"

    dataset = load_data(train_csv, test_csv)
    dataset = map_labels(dataset)
    model_name = "cardiffnlp/twitter-roberta-base-sentiment-latest"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with 3 labels (Positive, Mixed, Negative)
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Tokenize text for RoBERTa
    def preprocess(examples):
        return tokenizer(examples["review_text"], truncation=True, padding="max_length", max_length=256)

    dataset = dataset.map(preprocess, batched=True)
    
    # Compute class weights to handle imbalance
    train_labels = [example["label"] for example in dataset["train"]]
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.array([0, 1, 2]),
        y=np.array(train_labels)
    )
    
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    print(f"\nClass weights: {class_weights}")
    print(f"Train label distribution: {np.bincount(train_labels)}")

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned-accessibility-bert",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        logging_dir='./logs',
        logging_steps=50,
        warmup_steps=100
    )

    # Trainer with class weights
    trainer = WeightedTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        class_weights=class_weights
    )

    # Fine-tune
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained("./finetuned-accessibility-bert/final_model")
    tokenizer.save_pretrained("./finetuned-accessibility-bert/final_model")
    print(f"Fine-tuning complete. Model and tokenizer saved to ./finetuned-accessibility-bert/final_model")

if __name__ == "__main__":
    main()
