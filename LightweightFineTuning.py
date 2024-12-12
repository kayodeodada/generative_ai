# Lightweight Fine-Tuning for Sentiment Analysis using GPT-2
#
# Description of the approach:

# * PEFT technique: 
#   LoRA (Low-Rank Adaptation) is the selected PEFT techniques since it is compatible with all models
# * Model:
#   GPT-2 is the selected model since it is relatively small and compatible with sequence classification and LoRA
#
# * Evaluation approach: 
#
#   Evaluation will be conducted using Hugging Face's `Trainer` class which simplifies the training 
#   and evaluation workflow. accuracy and F1-score are computed using `compute_metrics` function which allaows for a comparison 
#   of the performance of the original model against the fine-tunned model
#
# * Fine-tuning dataset: 
#   The IMDb dataset from the Hugging Face `datasets` library will be used for fine-tunning. This dataset is a standard benchmark for binary sentiment classification tasks making it a good fit in this context.

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_base_model(model_name, num_labels):
    """ Load the base model and tokenizer for fine-tuning """
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def prepare_dataset(dataset_name, tokenizer):
    """ Load the IMDb dataset for fine-tuning """

    dataset = load_dataset(dataset_name)
    train_dataset = dataset["train"].shuffle(seed=42).select(range(1000))
    test_dataset = dataset["test"].shuffle(seed=42).select(range(500))

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True)

    train_dataset = train_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)
    return train_dataset, test_dataset

def compute_metrics(eval_pred):
    """ Define the compute_metrics function for the Trainer class """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    return {"accuracy": acc, "f1": f1}