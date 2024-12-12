## Project Introduction ##
Applying parameter-efficient fine-tuning using the Hugging Face `peft` library
Lightweight fine-tuning is important for adapting foundation models, because it allows you to modify foundation models for your needs without needing substantial computational resources.

This project brings together all of the essential components of a PyTorch + Hugging Face training and inference process. Specifically:
- Loading a pre-trained model and evaluate its performance
- Performing parameter-efficient fine tuning using the pre-trained model
- Performing inference using the fine-tuned model and compare its performance to the original model