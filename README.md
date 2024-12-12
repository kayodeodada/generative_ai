## Project Introduction ##
Applying parameter-efficient fine-tuning using the Hugging Face `peft` library
Lightweight fine-tuning is important for adapting foundation models, because it allows you to modify foundation models for your needs without needing substantial computational resources.

This project brings together all of the essential components of a PyTorch + Hugging Face training and inference process. Specifically:
- Loading a pre-trained model and evaluate its performance
- Performing parameter-efficient fine tuning using the pre-trained model
- Performing inference using the fine-tuned model and compare its performance to the original model

### Description of the approach ###

Lightweight Fine-Tuning for Sentiment Analysis using GPT-2

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