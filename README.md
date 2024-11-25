# MIMIC-CXR Data Analysis: Chest X-ray Image Classification and Report Generation

## Overview
This project aims to automatically classify abnormalities in chest X-ray images and generate text summaries that mirror radiologist reports using the MIMIC-CXR dataset. The goal is to enhance the speed and accuracy of interpreting chest X-rays by identifying common pathologies (e.g., pneumonia, pulmonary edema, pleural effusion) and summarizing them in a readable format.

## Project Goal
The objective is to combine computer vision and Natural Language Processing (NLP) to automatically generate diagnostic reports from chest X-rays. Specifically, we are focused on:
- Classifying pathologies in X-ray images.
- Generating accurate text summaries of the findings.

## Key Challenges
- **Image Classification**: Classifying multiple pathologies in chest X-ray images.
- **Natural Language Processing (NLP)**: Generating readable and clinically relevant text summaries from the classified images.
- **Dataset Complexity**: Handling a large and complex dataset with detailed annotations for numerous conditions.

## Technologies Used
- **Computer Vision**: Convolutional Neural Networks (CNN) such as ResNet or EfficientNet for image classification.
- **NLP Models**: XXX or XXX models for text summarization.
- **Multimodal Learning**: Techniques like CLIP for image-text alignment.
- **Evaluation Metrics**: 
  - For classification: AUC, precision, recall, F1-score.
  - For text generation: XXX,XXX.

## Project Approach
1. **Data Preprocessing**:
   - Clean the dataset, normalize images, resize them, and handle class imbalances.
2. **Model Training**:
   - Fine-tune a pre-trained CNN model (ResNet or EfficientNet) for pathologies classification.
   - Train NLP models (BERT or T5) for summarizing radiologist-like reports based on images and their associated labels.
3. **Multimodal Learning**:
   - Investigate the use of CLIP to improve the alignment between the image features and the text descriptions.
4. **Evaluation**:
   - Evaluate the performance of the models using appropriate metrics for classification and text generation.

## Deliverables
- **Image Classification Model**: A CNN that classifies pathologies in chest X-rays.
- **Text Summarization Model**: A model that generates readable diagnostic text summaries.
- **Final Report and Presentation**: A comprehensive report documenting methodology, results, and challenges, along with a presentation summarizing the project.

## Team Members
- **Yuandi Tang**: Focus on fine-tuning and evaluation of models, reporting.
- **Feng-Jen Hsieh**: Focus on image preprocessing and CNN model training.
- **Alexander Koehler**: Handle text preprocessing and NLP model training for report generation.

## Dataset
- **MIMIC-CXR**: A large, publicly available collection of de-identified chest X-ray images and corresponding radiology reports from the Beth Israel Deaconess Medical Center ICU. The dataset contains over 370,000 images and detailed labels for common findings (e.g., pneumonia, pleural effusion, cardiomegaly).
