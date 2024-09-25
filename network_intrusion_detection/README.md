# Network Intrusion Detection
This repository contains the code for the **Network Intrusion Detection** assignment as part of the "Machine Learning 
for Cybersecurity" course at Reykjavik University, Fall 2024. The goal of this project is to develop and evaluate models 
(Decision Trees, Random Forests) to detect and classify network attacks using machine learning techniques.

## Project Overview

This project focuses on detecting various types of network attacks by analyzing network traffic data. The dataset used contains features like packet sizes, flags, flow rates, and other indicators of potential network intrusions. The models developed (Decision Trees and Random Forests) aim to classify network traffic into the following classes:
- **Benign**: Normal traffic.
- **DoS (Denial of Service)**: Traffic meant to overwhelm and disrupt a service.
- **PortScan**: Traffic attempting to scan network ports for vulnerabilities.
- **Exploit**: Traffic exploiting known vulnerabilities to gain unauthorized access.

## Directory Structure

The project directory contains the following key files:

- [preprocess.py](preprocess.py): Preprocessing the dataset (cleaning, label aggregation, encoding)
- [model_trainer.py](model_trainer.py): Training and evaluating machine learning models 
- [resampler.py](resampler.py): Resampling the dataset (balanced undersampling/oversampling) 
- [pipeline.py](pipeline.py): The main pipeline to run preprocessing, model training, and evaluation 
- [resources/](resources): Stores output files like confusion matrix plots and feature importance plots

