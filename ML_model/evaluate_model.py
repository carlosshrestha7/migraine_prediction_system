# evaluate_model.py
"""
Model evaluation and performance analysis
"""

import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from migraine_model import MigrainePredictionModel
import numpy as np

def evaluate_model():
    # Load the trained model
    model_data = joblib.load('migraine_model.pkl')
    model = MigrainePredictionModel()
    model.occurrence_model = model_data['occurrence_model']
    model.scaler = model_data['scaler']
    
    # Load your data
    df = pd.read_csv('event_dump.csv')
    
    # Engineer features
    features = model.engineer_features(df)
    
    if 'migraine_occurrence' not in features.columns:
        print("❌ No target variable found in features")
        return
    
    X = features.drop('migraine_occurrence', axis=1)
    y = features['migraine_occurrence']
    
    # Scale features
    X_scaled = model.scaler.transform(X)
    
    # Make predictions
    y_pred = model.occurrence_model.predict(X_scaled)
    y_pred_proba = model.occurrence_model.predict_proba(X_scaled)[:, 1]
    
    # Print evaluation metrics
    print("📊 MODEL EVALUATION METRICS")
    print("=" * 50)
    print(f"Total samples: {len(y)}")
    print(f"Migraine cases: {y.sum()} ({y.sum()/len(y)*100:.2f}%)")
    print(f"Non-migraine cases: {len(y)-y.sum()} ({(len(y)-y.sum())/len(y)*100:.2f}%)")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Feature Importance
    print("\n🏆 TOP 10 FEATURE IMPORTANCES:")
    feature_importance = dict(zip(X.columns, model.occurrence_model.feature_importances_))
    for i, (feat, imp) in enumerate(sorted(feature_importance.items(), 
                                         key=lambda x: x[1], reverse=True)[:10], 1):
        print(f"{i:2d}. {feat:30s} {imp:.4f}")
    
    # ROC Curve (if you want to plot)
    if len(np.unique(y)) > 1:
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        print(f"\n📈 ROC-AUC Score: {roc_auc:.4f}")

if __name__ == "__main__":
    evaluate_model()
