# AI Explainability in Classifying Political Speeches and Interviews

This repository contains the code and resources for the paper "AI Explainability in Classifying Political Speeches and Interviews". The project applies explainable AI techniques to understand the linguistic features involved in classifying speeches and interviews in political discourse, using a feature-based Linguistic-Rule-Based Model (LRBM), logistic regression, Transformer-based models, and SHAP values.

## Project Overview

This study proposes a binary classification model that distinguishes speeches from interviews in political discourse. Speeches are unidirectional, allowing a speaker to address an audience without direct interruption (monologic), whereas interviews are bidirectional, marked by an interactive exchange between interviewer and interviewee (dialogic).

The research focuses on analyzing ten common linguistic features that differentiate speeches from interviews:
- Sentence Length
- Word Length
- Sentence Complexity
- Personal Pronoun Frequency
- Passive Voice Frequency
- Lexical Word Frequency
- Nominalization Frequency
- Interjection Frequency
- Modal Verb Frequency
- Discourse Marker Frequency

The project includes multiple datasets and models to analyze how these linguistic features influence the classification of political texts, with a focus on improving the explainability of BERT model predictions.

## Academic Context

This paper is part of ongoing research in the field of computational linguistics and political discourse analysis:

**"AI Explainability in Classifying Political Speeches and Interviews"**

This work was published in:
Reyes, J. F., AI Explainability in Classifying Political Speeches and Interviews, Journal of Language Technology and Computational Linguistics (JLCL). Under review.

## Hugging Face Resources

The models and datasets used in this project are published on Hugging Face:

- **Models**: 
  - Speech-vs-Interview-Classification-BERT, https://doi.org/10.57967/hf/2649
  - Speech-vs-Interview-Classification-BERT-Anonym, https://doi.org/10.57967/hf/2648
- **Datasets**: 
  - Speech-vs-Interview-Dataset, https://doi.org/10.57967/hf/2651
  - Speech-vs-Interview-Dataset-Anonym, https://doi.org/10.57967/hf/2650

## Repository Structure

### Root Directory Python Files

- **paper_a_1_dataset_1_build_raw.py**: Builds the raw dataset from collected political texts
- **paper_a_2_dataset_1a_preprocess.py**: Preprocesses the dataset for feature extraction
- **paper_a_3_dataset_1b_preprocess.py**: Preprocesses the dataset for BERT models
- **paper_a_4_dataset_1a_split.py**: Splits the feature dataset into train and test sets
- **paper_a_5_dataset_1b_split_sliding_window.py**: Splits the text dataset using a sliding window approach
- **paper_a_6_dataset_1a_feature_extraction.py**: Extracts linguistic features from the dataset
- **paper_a_7_dataset_1a_data_processing.py**: Processes the extracted features
- **paper_a_8_dataset_1a_t_test.py**: Performs t-test analysis on the features
- **paper_a_9_logistic_regression.py**: Implements logistic regression model for classification
- **paper_a_10_svm.py**: Implements SVM model for classification
- **paper_a_12_train_bert_better.py**: Trains the BERT model for speech vs. interview classification
- **paper_a_14_test_bert.py**: Evaluates the trained BERT model on the test dataset
- **paper_a_15_shap_analysis.py**: Performs SHAP analysis for model explainability
- **paper_a_20_inference_bert.py**: Performs inference using the trained BERT model

### Library Files (lib/)

- **lexicons.py**: Contains lexicons for linguistic feature extraction
- **linguistic_utils.py**: Provides utilities for linguistic analysis
- **measure_features.py**: Implements feature measurement functions
- **ner_processing.py**: Processes named entities for anonymization
- **text_utils.py**: Provides text processing utilities
- **transition_markers.py**: Contains discourse markers for analysis
- **utils.py**: Contains general utility functions used across the project
- **visualizations.py**: Implements visualization functions for analysis results

### Images (images/paper_a/)

- **paper_a_1_bert_confusion_matrix.png**: Confusion matrix for BERT model
- **paper_a_1_bert_losses.png**: Training losses for BERT model
- **paper_a_7_correlation_matrix_heatmap.png**: Correlation matrix of linguistic features
- **paper_a_11_interjection_histogram_and_box_plot.png**: Analysis of interjection frequency
- **paper_a_20_speech_shap_features_plot_bar.png**: SHAP values for speech classification
- **paper_a_21_discourse_marker_freq_boxplots.png**: Analysis of discourse marker frequency

### Datasets (shared_data/)

- **dataset_1_1_raw.jsonl**: Raw dataset with political speeches and interviews
- **dataset_1_2_1a_preprocessed.jsonl**: Preprocessed dataset for feature extraction
- **dataset_1_4_1a_train.jsonl**: Training dataset with features
- **dataset_1_4_1a_test.jsonl**: Test dataset with features
- **dataset_1_6_1b_train.jsonl**: Training dataset for BERT models
- **dataset_1_6_1b_test.jsonl**: Test dataset for BERT models
- **dataset_1_6_1b_validation.jsonl**: Validation dataset for BERT models
- **dataset_1_6_1b_train_anonym.jsonl**: Anonymized training dataset for BERT models
- **dataset_1_6_1b_test_anonym.jsonl**: Anonymized test dataset for BERT models
- **dataset_1_6_1b_validation_anonym.jsonl**: Anonymized validation dataset for BERT models

### Documentation

- **paper-a.html**: The full research paper describing the methodology and findings

## Usage

1. **Dataset Preparation**: 
   - Run `paper_a_1_dataset_1_build_raw.py` to build the raw dataset
   - Run `paper_a_2_dataset_1a_preprocess.py` and `paper_a_3_dataset_1b_preprocess.py` to preprocess the data
   - Run `paper_a_4_dataset_1a_split.py` and `paper_a_5_dataset_1b_split_sliding_window.py` to split the datasets

2. **Feature Extraction**:
   - Run `paper_a_6_dataset_1a_feature_extraction.py` to extract linguistic features
   - Run `paper_a_7_dataset_1a_data_processing.py` to process the features
   - Run `paper_a_8_dataset_1a_t_test.py` to analyze feature significance

3. **Model Training**:
   - Run `paper_a_9_logistic_regression.py` to train the logistic regression model
   - Run `paper_a_12_train_bert_better.py` to train the BERT model

4. **Model Evaluation**:
   - Run `paper_a_14_test_bert.py` to evaluate the BERT model
   - Run `paper_a_15_shap_analysis.py` to perform explainability analysis

5. **Inference**:
   - Run `paper_a_20_inference_bert.py` to perform inference with the trained models

## Requirements

The project dependencies are listed in the `requirements.txt` file. Install them using:

```
pip install -r requirements.txt
```

Key dependencies include:
- transformers==4.25.1
- numpy==1.26.2
- torch==2.1.2
- spacy==3.4.4
- matplotlib==3.8.2
- oauth2client==4.1.3
- scipy==1.11.4
- optuna==3.5.0
- unidecode==1.3.8
- statsmodels==0.14.1

## Research Findings

The analysis reveals four key "political discourse features" that distinguish speeches from interviews:
1. **Nominalization Frequency**: Higher in speeches, contributing to their formal and abstract nature
2. **Discourse Marker Frequency**: Higher in interviews, facilitating interactive communication
3. **Personal Pronoun Frequency**: Higher in interviews, managing conversational dynamics
4. **Interjection Frequency**: Higher in interviews, expressing attitudes and emotional reactions

The study demonstrates that BERT models rely on both linguistic structures and thematic context for classification, with anonymization shifting the focus toward linguistic features. SHAP analysis provides insights into which features most strongly influence the model's predictions, enhancing the explainability of the classification process.

## Citation

If you use this code or the findings in your research, please cite the original paper:

```
Reyes, J. F., AI Explainability in Classifying Political Speeches and Interviews, Journal of Language Technology and Computational Linguistics (JLCL). Under review.
```
