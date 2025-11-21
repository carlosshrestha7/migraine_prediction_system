# migraine_model.py
"""
Migraine prediction ML model - UPDATED for your data structure
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')


class MigrainePredictionModel:
    """
    ML Model for predicting migraine occurrence - UPDATED for your CSV structure
    """

    def __init__(self):
        self.occurrence_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

    def _convert_to_numeric(self, series, default=0):
        """Safely convert series to numeric, handling strings and edge cases"""
        if series.dtype == 'bool':
            return series.astype(int)

        numeric_series = pd.to_numeric(series, errors='coerce')
        return numeric_series.fillna(default)

    def _get_column(self, df, *possible_names, default=0):
        """Try multiple column name variations and return the first match"""
        for name in possible_names:
            if name in df.columns:
                return self._convert_to_numeric(df[name], default)
        return pd.Series([default] * len(df))

    def engineer_features(self, df, gender='female'):
        """
        Create engineered features from YOUR actual CSV structure
        """
        features = pd.DataFrame()

        # Time-based features
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], errors='coerce')
            features['hour_of_day'] = df['datetime'].dt.hour.fillna(12)
            features['day_of_week'] = df['datetime'].dt.dayofweek.fillna(0)
            features['is_weekend'] = (features['day_of_week'] >= 5).astype(int)
        else:
            features['hour_of_day'] = 12
            features['day_of_week'] = 0
            features['is_weekend'] = 0

        # Core features from YOUR dataset
        # Stress features
        features['stress_level'] = self._get_column(df, 'stressLevel.value', 'predictedStress', 'originalPredictedStress', default=5)
        features['high_stress'] = (features['stress_level'] > 7).astype(int)
        
        # Mood features
        features['mood_value'] = self._get_column(df, 'mood.value', default=5)
        features['low_mood'] = (features['mood_value'] < 4).astype(int)
        
        # Food intake features
        features['skipped_breakfast'] = self._get_column(df, 'foodIntake.skippedBreakfast', 'foodIntake.hadBreakfast', default=0)
        features['skipped_lunch'] = self._get_column(df, 'foodIntake.skippedLunch', 'foodIntake.hadLunch', default=0)
        features['skipped_dinner'] = self._get_column(df, 'foodIntake.skippedDinner', 'foodIntake.hadDinner', default=0)
        features['total_meals_skipped'] = features['skipped_breakfast'] + features['skipped_lunch'] + features['skipped_dinner']
        features['missed_meals'] = (features['total_meals_skipped'] > 0).astype(int)
        
        # Activity features
        features['activity_index'] = self._get_column(df, 'activity_index', default=50)
        features['sedentary'] = (features['activity_index'] < 30).astype(int)
        
        # Migraine history features (using available columns)
        features['has_symptoms'] = self._get_column(df, 'symptoms', default=0)
        features['has_triggers'] = self._get_column(df, 'triggers', default=0)
        features['migraine_intensity'] = self._get_column(df, 'intensity', default=0)
        features['took_medication'] = self._get_column(df, 'tookMedication', default=0)
        
        # Duration features
        features['duration_minutes'] = self._get_column(df, 'duration_m', default=0)
        
        # Prediction features
        features['predicted_stress'] = self._get_column(df, 'predictedStress', default=5)
        features['original_predicted_stress'] = self._get_column(df, 'originalPredictedStress', default=5)
        
        # Create migraine target variable from available data
        # If we have intensity > 0 or symptoms present, consider it a migraine
        features['migraine_occurrence'] = (
            (features['migraine_intensity'] > 0) | 
            (features['has_symptoms'] > 0)
        ).astype(int)

        # Interaction features
        features['stress_mood_interaction'] = features['stress_level'] * (10 - features['mood_value'])
        features['stress_meals_interaction'] = features['stress_level'] * features['total_meals_skipped']
        features['mood_activity_interaction'] = features['mood_value'] * features['activity_index']

        # Time-of-day features
        features['is_morning'] = ((features['hour_of_day'] >= 6) & (features['hour_of_day'] < 12)).astype(int)
        features['is_afternoon'] = ((features['hour_of_day'] >= 12) & (features['hour_of_day'] < 18)).astype(int)
        features['is_evening'] = ((features['hour_of_day'] >= 18) | (features['hour_of_day'] < 6)).astype(int)

        # Fill any remaining NaN values
        features = features.fillna(0)

        # Ensure all columns are numeric
        for col in features.columns:
            features[col] = self._convert_to_numeric(features[col])

        print(f"🔧 Engineered {len(features.columns)} features from your data")
        return features

    def train(self, df, gender='female', sample_size=None):
        """Train the migraine prediction models"""

        # Sample data if too large
        if sample_size and len(df) > sample_size:
            print(f"📊 Sampling {sample_size} records from {len(df)} total records...")
            df = df.sample(n=sample_size, random_state=42)

        print("Engineering features...")
        features = self.engineer_features(df, gender)

        # Use the engineered migraine_occurrence as target
        if 'migraine_occurrence' not in features.columns:
            raise ValueError("Could not create migraine target variable from available data!")

        y_occurrence = features['migraine_occurrence']
        
        # Remove the target from features
        X = features.drop('migraine_occurrence', axis=1)

        # Remove invalid rows
        valid_idx = (y_occurrence.notna()) & (X.notna().all(axis=1))

        if valid_idx.sum() == 0:
            raise ValueError("No valid training data found. Please check your CSV file.")

        X = X[valid_idx]
        y_occurrence = y_occurrence[valid_idx]

        print(f"\n✅ Training with {len(X)} valid samples")
        print(f"📊 Features used: {len(X.columns)}")
        print(f"🔴 Migraine occurrences: {y_occurrence.sum():,} ({y_occurrence.sum() / len(y_occurrence) * 100:.2f}%)")
        print(f"🟢 Non-migraine days: {(~y_occurrence.astype(bool)).sum():,} ({(~y_occurrence.astype(bool)).sum() / len(y_occurrence) * 100:.2f}%)")

        if y_occurrence.sum() < 10:
            print("⚠️  WARNING: Very few migraine occurrences in data. Model may not be reliable.")

        # Split data
        test_size = 0.2
        stratify_param = y_occurrence if y_occurrence.sum() > 5 else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_occurrence, test_size=test_size, random_state=42, stratify=stratify_param
        )

        # Scale features
        print("\n🔧 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("🤖 Training occurrence model...")
        self.occurrence_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        self.occurrence_model.fit(X_train_scaled, y_train)

        # Evaluate
        print("\n" + "=" * 60)
        print("MODEL EVALUATION")
        print("=" * 60)

        y_pred = self.occurrence_model.predict(X_test_scaled)
        y_pred_proba = self.occurrence_model.predict_proba(X_test_scaled)[:, 1]

        print(f"\n📈 Accuracy: {accuracy_score(y_test, y_pred):.4f}")

        if y_test.sum() > 0:  # Only calculate AUC if we have positive samples
            print(f"📊 ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Feature importance
        feature_names = X.columns
        importances = self.occurrence_model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importances))

        print("\n🏆 Top 15 Most Important Features:")
        for i, (feat, imp) in enumerate(sorted(self.feature_importance.items(),
                                               key=lambda x: x[1], reverse=True)[:15], 1):
            bar = "█" * int(imp * 100)
            print(f"  {i:2d}. {feat:30s} {imp:6.4f} {bar}")

        print("=" * 60)
        return self

    def predict(self, user_data, gender='female'):
        """
        Predict migraine occurrence probability
        """
        # Engineer features
        features = self.engineer_features(user_data, gender)
        
        # Ensure we have features to work with
        if len(features) == 0:
            return self._get_default_prediction()
        
        # Remove target column if present
        if 'migraine_occurrence' in features.columns:
            features = features.drop('migraine_occurrence', axis=1)
            
        X_scaled = self.scaler.transform(features)

        # Predict occurrence probability
        occurrence_prob = self.occurrence_model.predict_proba(X_scaled)[0][1]

        # Calculate risk band
        if occurrence_prob < 0.20:
            risk_band = "green"
            risk_label = "Low Risk"
            risk_message = "Low probability of migraine"
        elif occurrence_prob < 0.50:
            risk_band = "yellow"
            risk_label = "Moderate Risk"
            risk_message = "Moderate probability of migraine"
        elif occurrence_prob < 0.70:
            risk_band = "orange"
            risk_label = "High Risk"
            risk_message = "High probability of migraine"
        else:
            risk_band = "red"
            risk_label = "Very High Risk"
            risk_message = "Very high probability of migraine"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            features.iloc[0], occurrence_prob, gender
        )

        return {
            'probability': round(occurrence_prob * 100, 1),
            'risk_band': risk_band,
            'risk_label': risk_label,
            'risk_message': risk_message,
            'recommendations': recommendations,
            'top_risk_factors': self._identify_risk_factors(features.iloc[0])
        }

    def _get_default_prediction(self):
        """Return default prediction when no features are available"""
        return {
            'probability': 10.0,
            'risk_band': 'green',
            'risk_label': 'Low Risk',
            'risk_message': 'Insufficient data for accurate prediction',
            'recommendations': [],
            'top_risk_factors': ['Insufficient data']
        }

    def _identify_risk_factors(self, features):
        """Identify current risk factors"""
        risk_factors = []

        if features.get('stress_level', 0) > 7:
            risk_factors.append("High stress level")
        if features.get('total_meals_skipped', 0) > 0:
            risk_factors.append("Missed meals")
        if features.get('low_mood', 0) > 0:
            risk_factors.append("Low mood")
        if features.get('sedentary', 0) > 0:
            risk_factors.append("Low activity level")
        if features.get('high_stress', 0) > 0:
            risk_factors.append("Elevated stress")

        return risk_factors

    def _generate_recommendations(self, features, probability, gender):
        """Generate personalized recommendations"""
        recommendations = []

        # Stress management
        if features.get('stress_level', 0) > 7:
            recommendations.append({
                'category': 'Stress Management',
                'action': 'Your stress level is high. Practice deep breathing or meditation',
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        # Meal planning
        if features.get('missed_meals', 0) > 0:
            recommendations.append({
                'category': 'Nutrition',
                'action': 'Avoid skipping meals. Have regular, balanced meals',
                'priority': 'high'
            })

        # Activity recommendations
        if features.get('sedentary', 0) > 0:
            recommendations.append({
                'category': 'Activity',
                'action': 'Light exercise or a short walk may help reduce stress',
                'priority': 'medium'
            })

        # Mood management
        if features.get('low_mood', 0) > 0:
            recommendations.append({
                'category': 'Mood',
                'action': 'Take breaks and engage in enjoyable activities',
                'priority': 'medium'
            })

        # Preventive measures for high risk
        if probability > 0.6:
            recommendations.append({
                'category': 'Prevention',
                'action': 'Consider taking preventive measures and stay hydrated',
                'priority': 'high'
            })

        # Hydration (always include)
        recommendations.append({
            'category': 'Hydration',
            'action': 'Ensure you stay well hydrated throughout the day',
            'priority': 'medium' if probability > 0.4 else 'low'
        })

        return recommendations

    def save_model(self, filepath='migraine_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'occurrence_model': self.occurrence_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")

    def load_model(self, filepath='migraine_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.occurrence_model = model_data['occurrence_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        print(f"✅ Model loaded from {filepath}")
        return self


class ModelTrainer:
    """Handles the complete training pipeline"""

    @staticmethod
    def load_csv_data(filepath):
        """Load and preprocess CSV data"""
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
        print(f"📋 Columns found: {len(df.columns)} columns")
        return df

    @staticmethod
    def validate_data(df):
        """Validate data quality and show statistics"""
        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)
        print(f"Total records: {len(df):,}")

        # Check for important columns
        important_cols = ['stressLevel.value', 'mood.value', 'foodIntake.skippedBreakfast', 'intensity']
        available_cols = [col for col in important_cols if col in df.columns]
        print(f"✅ Available important columns: {available_cols}")

        # Data completeness
        print(f"\n📉 Missing Data (top 10 columns with most missing data):")
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        for col, pct in list(missing.items())[:10]:
            if pct > 0:
                print(f"   {col}: {pct:.1f}%")

        print("=" * 60)
        return True

    @staticmethod
    def train_model_from_csv(csv_filepath, gender='female', save_path='migraine_model.pkl', sample_size=None):
        """Complete training pipeline from CSV file"""
        # Load data
        df = ModelTrainer.load_csv_data(csv_filepath)

        # Validate
        ModelTrainer.validate_data(df)

        # Train model
        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        model = MigrainePredictionModel()
        model.train(df, gender=gender, sample_size=sample_size)

        # Save model
        model.save_model(save_path)

        return model


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 MIGRAINE PREDICTION ML MODEL - TRAINING MODULE")
    print("=" * 60 + "\n")

    # Train with your CSV file
    model = ModelTrainer.train_model_from_csv(
        csv_filepath='event_dump.csv',
        gender='female',
        save_path='migraine_model.pkl'
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("📦 Model saved and ready for predictions.")
    print("💡 Use model.predict(new_data) to make predictions.")
    print("=" * 60)