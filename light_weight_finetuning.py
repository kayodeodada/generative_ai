"""
 Lightweight Fine-Tuning for Sentiment Analysis using GPT-2
Description of the approach:

* PEFT technique:
  LoRA (Low-Rank Adaptation) is the selected PEFT techniques since it is compatible with all models
* Model: 
    GPT-2 is the selected model since it is relatively small 
    and compatible with sequence classification and LoRA
* Evaluation approach:

    Evaluation will be conducted using Hugging Face's `Trainer` class which simplifies the training
     and evaluation workflow. accuracy and F1-score are computed using `compute_metrics` function 
    which allaows for a comparison
    of the performance of the original model against the fine-tunned model

* Fine-tuning dataset:
  The IMDb dataset from the Hugging Face `datasets` library will be used for fine-tunning. 
  This dataset is a standard benchmark for binary s
  entiment classification tasks making it a good fit in this context.
"""
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, PeftModelForSequenceClassification
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

def load_base_model(model_name, num_labels):
    """ Load the base model and tokenizer for fine-tuning """
    model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1})

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

def create_trainer(model, train_dataset, test_dataset):
    """ Train the model using the Trainer class """
    # Define the training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        eval_strategy="epoch",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=10,
    )

    # Create the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics
    )

    return trainer

def create_peft_model(model):
    """ Create a PEFT model using the LoRA technique """
    # Define LoRA configuration
    lora_config = LoraConfig(
        task_type="SEQ_CLS", ## SEQ_CLS for sequence classification tasks
        inference_mode=False,
        lora_dropout=0.1
    )

    # Create PEFT model
    peft_model = get_peft_model(model, lora_config)

    return peft_model

def load_saved_model(model, output_dir, test_dataset):
    """ Load the saved fine-tuned model """
    # Reload the PEFT model
    loaded_model = PeftModelForSequenceClassification.from_pretrained(model,output_dir)
    loaded_model.eval()

    # Create trainer for evaluation
    loaded_model_trainer = create_trainer(
        model=loaded_model,
        train_dataset=None,
        test_dataset=test_dataset)

    return loaded_model_trainer.evaluate(), loaded_model

def predictions(model, tokenizer_inputs):
    """ Make predictions using the fine-tuned model """

    # Move model and inputs to device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    inputs = {k: v.to(device) for k, v in tokenizer_inputs.items()}

    # Perform inference
    outputs = model(**inputs)
    predicted_class = np.argmax(outputs.logits.detach().numpy(), axis=-1)

    return predicted_class
