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