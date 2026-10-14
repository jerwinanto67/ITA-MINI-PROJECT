# CropSense

CropSense is a machine learning-based crop recommendation web application built with Python and Streamlit. It uses soil nutrients (Nitrogen, Phosphorus, Potassium), soil pH, and climate data (temperature, humidity, rainfall) to suggest the most suitable crop using models trained on the Kaggle Crop Recommendation dataset.

## Project Structure

```text
├── data/
│   └── Crop_recommendation.csv    # Kaggle dataset
├── model/
│   └── crop_model.pkl             # Trained serialized model
├── outputs/
│   ├── feature_distribution.png   # Input feature histograms
│   ├── correlation_heatmap.png    # Feature correlation heatmap
│   ├── model_comparison.png       # Test accuracy comparison chart
│   ├── confusion_matrix.png       # Confusion matrix of best model
│   ├── feature_importance.png     # Feature importance bar chart
│   └── results.json               # Detailed evaluation metrics
├── app.py                         # Streamlit web application
├── train_model.py                 # Training and evaluation script
├── requirements.txt               # Dependencies
└── README.md                      # Documentation
```

## Setup & Execution Instructions

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Train and Compare Models**
   ```bash
   python train_model.py
   ```
   This script compares Decision Tree, Naive Bayes, Logistic Regression, and Random Forest, performs 5-fold cross validation, saves the best model to `model/crop_model.pkl`, and exports charts and metrics to the `outputs/` folder.

3. **Launch the Streamlit App**
   ```bash
   streamlit run app.py
   ```
   Or if `streamlit` is not directly in PATH:
   ```bash
   python -m streamlit run app.py
   ```
