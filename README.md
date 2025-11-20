# Sensor2Robot
Machine Learning on SECOM sensor data for anomaly detection and integration with RoboDK for smart manufacturing automation.
# RoboSECOM

## Overview
This project uses SECOM sensor data to predict product quality (pass/fail) using machine learning models and integrates the prediction results with RoboDK for robot control in manufacturing automation.

## Features
- Load and preprocess SECOM dataset
- Train and compare multiple ML models (Logistic Regression, Random Forest, XGBoost)
- Predict anomalies in real-time sensor data
- Send commands to RoboDK robots based on predictions

## Dataset
- SECOM dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/179/secom)

## Requirements
- Python 3.x
- pandas, scikit-learn, xgboost, robolink, robodk
