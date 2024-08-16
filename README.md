# 🖼️ Image Captioning with Fine-Tuned ViT and GPT-2

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Hugging%20Face-FFD21E?style=for-the-badge&logo=huggingface&logoColor=000" alt="Hugging Face Transformers">
  <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="scikit-learn">
  <img src="https://img.shields.io/badge/COCO%20Dataset-009688.svg?style=for-the-badge&logo=OpenCV&logoColor=white" alt="COCO Dataset">
  <img src="https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54" alt="Python">
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" alt="Jupyter Notebook">
</p>

Welcome to the **Image Captioning** project! This repository implements an advanced image captioning module that leverages state-of-the-art models, including the **ViT-Base-Patch16-224-In21k** (Vision Transformer) as the encoder and **DistilGPT-2** as the decoder. This project aims to generate descriptive captions for images from the COCO dataset, utilizing the powerful capabilities of the Transformers library.

---

## 📝 Project Description

This project focuses on creating an image captioning system by integrating the following key components:

- **Encoder**: The project uses the Google **ViT-Base-Patch16-224-In21k** pretrained model to encode image features. ViT (Vision Transformer) is known for its superior performance in image classification and feature extraction tasks.
- **Decoder**: The **DistilGPT-2** model, a distilled version of GPT-2, is employed to decode the image features into natural language captions. GPT-2 excels at generating coherent and contextually relevant text.

### 🎯 Objective

The primary goal is to fine-tune these models on the COCO dataset for the image captioning task. The resulting captions are evaluated using popular NLP metrics like **ROUGE**, **BLEU**, and **BERTScore** to measure their quality and relevance.

---

## 📚 Dataset

The project utilizes the **COCO dataset** (Common Objects in Context), which is a rich dataset consisting of:

- **118,000** training images
- **5,000** validation images
- Each image is paired with **5 corresponding captions**, providing diverse descriptions of the visual content.

This dataset is well-suited for training and evaluating image captioning models due to its variety and scale.

---

## ⚙️ Implementation Details

### Frameworks & Libraries

- **PyTorch**: The deep learning framework used for model implementation and training.
- **Transformers**: Hugging Face's library is employed to access and fine-tune the ViT and GPT-2 models.

### Model Architecture

- **Vision Transformer (ViT)**: Acts as the encoder, transforming images into feature-rich embeddings.
- **DistilGPT-2**: Serves as the decoder, generating textual descriptions based on the encoded image features.

### Training Process

- **Fine-Tuning**: Both models are fine-tuned on the COCO dataset over **2 epochs**. This process adapts the pretrained models to the specific task of image captioning, optimizing their performance on this task.

---

## 🧪 Evaluation Metrics

The generated captions are evaluated using the following metrics:

- **ROUGE**: Measures the overlap between the predicted and reference captions.
- **BLEU**: Evaluates the precision of n-grams in the generated captions compared to reference captions.
- **BERTScore**: Uses BERT embeddings to assess the semantic similarity between generated
