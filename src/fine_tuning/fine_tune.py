# src/fine_tuning/fine_tune.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig, Trainer, TrainingArguments
from datasets import load_dataset
import os

def main():
    # Dataset paths
    train_csv = "data/train_reviews.csv"
    test_csv = "data/test_reviews.csv"

    # Load dataset
    dataset = load_dataset("csv", data_files={"train": train_csv, "test": test_csv})

    # Map labels to numbers
    label_map = {"Positive": 0, "Mixed": 1, "Negative": 2}
    def map_labels(example):
        example["label"] = label_map[example["label"]]
        return example
    dataset = dataset.map(map_labels)

    # Load tokenizer
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with 3 labels, ignore mismatch
    config = AutoConfig.from_pretrained(model_name, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        config=config,
        ignore_mismatched_sizes=True
    )

    # Tokenize
    def preprocess(example):
        return tokenizer(example["review_text"], truncation=True, padding="max_length", max_length=128)
    dataset = dataset.map(preprocess, batched=True)

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./finetuned-accessibility-bert",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        logging_dir='./logs',
        save_steps=100
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"]
    )

    # Fine-tune
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained("./finetuned-accessibility-bert/final_model")
    tokenizer.save_pretrained("./finetuned-accessibility-bert/final_model")
    print(f"Fine-tuning complete. Model and tokenizer saved to ./finetuned-accessibility-bert/final_model")

if __name__ == "__main__":
    main()
