# evaluate_model.py
"""
Model evaluation and performance analysis
"""

import pandas as pd
import joblib
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from migraine_model import MigrainePredictionModel

def evaluate_model():
    """Evaluate the trained model performance"""
    
    # Check if model exists
    if not os.path.exists('migraine_model.pkl'):
        print("❌ Model file 'migraine_model.pkl' not found!")
        print("Please train the model first using train_model.py")
        return
    
    # Check if data exists
    data_file = 'event_dump.csv'
    if not os.path.exists(data_file):
        print(f"❌ Data file '{data_file}' not found!")
        return
    
    try:
        # Load the trained model
        print("📊 Loading model for evaluation...")
        model_data = joblib.load('migraine_model.pkl')
        model = MigrainePredictionModel()
        model.occurrence_model = model_data['occurrence_model']
        model.scaler = model_data['scaler']
        
        # Load your data
        print("📂 Loading data...")
        df = pd.read_csv(data_file)
        
        # Engineer features
        print("🔧 Engineering features...")
        features = model.engineer_features(df)
        
        if 'migraine_occurrence' not in features.columns:
            print("❌ No target variable found in features")
            return
        
        X = features.drop('migraine_occurrence', axis=1)
        y = features['migraine_occurrence']
        
        # Scale features
        X_scaled = model.scaler.transform(X)
        
        # Make predictions
        print("🎯 Making predictions...")
        y_pred = model.occurrence_model.predict(X_scaled)
        y_pred_proba = model.occurrence_model.predict_proba(X_scaled)[:, 1]
        
        # Print evaluation metrics
        print("\n" + "=" * 60)
        print("📊 MODEL EVALUATION METRICS")
        print("=" * 60)
        print(f"Total samples: {len(y):,}")
        print(f"Migraine cases: {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
        print(f"Non-migraine cases: {len(y)-y.sum():,} ({(len(y)-y.sum())/len(y)*100:.2f}%)")
        
        print("\n📋 Classification Report:")
        print(classification_report(y, y_pred, target_names=['No Migraine', 'Migraine']))
        
        # Confusion Matrix
        cm = confusion_matrix(y, y_pred)
        print("\n🎯 Confusion Matrix:")
        print(cm)
        
        # Calculate metrics from confusion matrix
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n📈 Detailed Metrics:")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        # Feature Importance
        print("\n🏆 TOP 15 FEATURE IMPORTANCES:")
        feature_importance = dict(zip(X.columns, model.occurrence_model.feature_importances_))
        for i, (feat, imp) in enumerate(sorted(feature_importance.items(), 
                                             key=lambda x: x[1], reverse=True)[:15], 1):
            print(f"{i:2d}. {feat:35s} {imp:.4f}")
        
        # ROC Curve and AUC
        if len(np.unique(y)) > 1:
            fpr, tpr, _ = roc_curve(y, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            print(f"\n📊 ROC-AUC Score: {roc_auc:.4f}")
            
            # Plot ROC curve
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            print("📈 ROC curve saved as 'roc_curve.png'")
        
        # Plot feature importance
        top_features = dict(sorted(feature_importance.items(), 
                                 key=lambda x: x[1], reverse=True)[:10])
        
        plt.figure(figsize=(10, 6))
        features_names = list(top_features.keys())
        importance_values = list(top_features.values())
        
        y_pos = np.arange(len(features_names))
        plt.barh(y_pos, importance_values, align='center', alpha=0.7, color='skyblue')
        plt.yticks(y_pos, features_names)
        plt.xlabel('Feature Importance')
        plt.title('Top 10 Most Important Features')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        print("📊 Feature importance plot saved as 'feature_importance.png'")
        
        print("\n" + "=" * 60)
        print("✅ EVALUATION COMPLETE!")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Evaluation failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    evaluate_model()