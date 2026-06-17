# Scope 3 Emissions Prediction Model

A machine learning-powered Scope 3 emissions estimation tool that predicts indirect greenhouse gas emissions using basic company operating metrics. The project combines a Random Forest regression model exported to ONNX format with a browser-based interface that performs inference entirely on the client side.

## Overview

Many organizations lack complete emissions disclosure data, particularly for Scope 3 emissions across their value chains. This project demonstrates how machine learning can estimate Scope 3 emissions from commonly available business information such as revenue, cost of goods sold (COGS), employee count, and sector classification.

The model is trained on synthetic corporate data and exported to ONNX, enabling deployment directly in the browser without requiring a backend server.

## Features

- Predicts Scope 3 emissions (metric tons CO2e)
- Browser-based inference using ONNX Runtime Web
- No server required for predictions
- Interactive user interface
- Sector-specific emission intensity adjustments
- Climate-risk visualization and emissions intensity indicators
- Portable ONNX model deployment

## Model Inputs

- Revenue_USD
- COGS_USD
- Employee_Count
- COGS_to_Revenue_Ratio
- Revenue_per_Employee
- Sector Classification

Supported sectors:
- Energy
- Technology
- Manufacturing
- Financials

## Machine Learning Pipeline

### Data Generation

The training dataset consists of 1,000 synthetic companies generated using realistic distributions for revenue, cost structure, employee count, and industry sector.

### Model

RandomForestRegressor with 100 trees and max depth of 8.

### Evaluation Metrics

- RMSE
- MAE
- R² Score

## Project Structure

Scope-3_Emissions_Prediction_Model/
├── index.html
├── train_export.py
├── scope3_model.onnx
└── README.md

## Running the Project

Run a local server:

python -m http.server 8000

Then open:
http://localhost:8000

## Technology Stack

- Python
- NumPy
- Pandas
- Scikit-Learn
- ONNX
- ONNX Runtime Web
- HTML
- CSS
- JavaScript

## Future Improvements

- Train on real ESG datasets
- Support more industries
- Add uncertainty estimation
- Build enterprise ESG dashboards

## Author

Omer Butt
Economics Student | Sustainability & Climate Analytics Enthusiast