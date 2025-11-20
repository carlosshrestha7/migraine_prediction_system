import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error
import joblib
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')


class MigrainePredictionModel:
    """
    ML Model for predicting migraine occurrence, intensity, and duration
    with gender-specific feature weighting
    """

    def __init__(self):
        self.occurrence_model = None
        self.intensity_model = None
        self.duration_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

        # Feature weights by category (adjustable based on research)
        self.feature_weights = {
            'sleep': 0.25,
            'stress': 0.20,
            'food_intake': 0.15,
            'weather': 0.10,
            'hormonal': 0.15,  # Higher for females
            'mood': 0.10,
            'activity': 0.05
        }

    def _convert_to_numeric(self, series, default=0):
        """Safely convert series to numeric, handling strings and edge cases"""
        # Convert boolean to int
        if series.dtype == 'bool':
            return series.astype(int)

        # Try to convert to numeric, coercing errors to NaN
        numeric_series = pd.to_numeric(series, errors='coerce')

        # Fill NaN with default value
        return numeric_series.fillna(default)

    def engineer_features(self, df, gender='female'):
        """
        Create engineered features from raw data
        """
        features = pd.DataFrame()

        # Time-based features
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='ms', errors='coerce')
            features['hour_of_day'] = df['datetime'].dt.hour.fillna(0)
            features['day_of_week'] = df['datetime'].dt.dayofweek.fillna(0)
            features['day_of_month'] = df['datetime'].dt.day.fillna(0)

        # Food intake patterns (15% weight)
        features['meals_skipped'] = (
                self._convert_to_numeric(df.get('foodIntake.skippedBreakfast', 0)) +
                self._convert_to_numeric(df.get('foodIntake.skippedDinner', 0)) +
                self._convert_to_numeric(df.get('foodIntake.skippedLunch', 0))
        )
        features['total_meals'] = (
                self._convert_to_numeric(df.get('foodIntake.hadBreakfast', 0)) +
                self._convert_to_numeric(df.get('foodIntake.hadDinner', 0)) +
                self._convert_to_numeric(df.get('foodIntake.hadLunch', 0))
        )
        features['meal_regularity_score'] = features['total_meals'] / 3

        # Stress indicators (20% weight)
        features['stress_level'] = self._convert_to_numeric(df.get('stressLevel.value', 0))
        features['stress_provided'] = self._convert_to_numeric(df.get('stressLevel.isProvided', 0))
        features['predicted_stress'] = self._convert_to_numeric(df.get('predictedStress', 0))
        features['original_predicted_stress'] = self._convert_to_numeric(df.get('originalPredictedStress', 0))
        features['stress_delta'] = abs(
            features['predicted_stress'] - features['original_predicted_stress']
        )

        # Mood indicators (10% weight)
        features['mood_value'] = self._convert_to_numeric(df.get('mood.value', 0))
        features['mood_provided'] = self._convert_to_numeric(df.get('mood.isProvided', 0))

        # Activity patterns (5% weight)
        features['activity_index'] = self._convert_to_numeric(df.get('activity_index', 0))
        features['sedentary_confirmed'] = self._convert_to_numeric(df.get('sedentaryConfirmed', 0))
        features['duration_minutes'] = self._convert_to_numeric(df.get('duration_m', 0))

        # Handle 'level' field if it exists (might be 'low', 'medium', 'high')
        if 'level' in df.columns:
            level_map = {'low': 1, 'medium': 2, 'high': 3}
            features['level_numeric'] = df['level'].map(level_map).fillna(0)

        # Historical migraine features
        if 'intensity' in df.columns:
            features['prev_intensity'] = self._convert_to_numeric(df['intensity'])
            features['had_migraine'] = (features['prev_intensity'] > 0).astype(int)

        # Symptom and trigger counts
        if 'symptoms' in df.columns:
            features['symptom_count'] = df['symptoms'].fillna('').astype(str).apply(
                lambda x: len([s for s in str(x).split(',') if s.strip()]) if x else 0
            )
        else:
            features['symptom_count'] = 0

        if 'triggers' in df.columns:
            features['trigger_count'] = df['triggers'].fillna('').astype(str).apply(
                lambda x: len([s for s in str(x).split(',') if s.strip()]) if x else 0
            )
        else:
            features['trigger_count'] = 0

        if 'extraSymptoms' in df.columns:
            features['extra_symptom_count'] = df['extraSymptoms'].fillna('').astype(str).apply(
                lambda x: len([s for s in str(x).split(',') if s.strip()]) if x else 0
            )
        else:
            features['extra_symptom_count'] = 0

        # Medication features
        if 'tookMedication' in df.columns:
            features['took_medication'] = self._convert_to_numeric(df['tookMedication'])
        if 'medicationWorked' in df.columns:
            features['medication_worked'] = self._convert_to_numeric(df['medicationWorked'])

        # Gender-specific features
        features['gender_female'] = 1 if gender.lower() == 'female' else 0

        # Interaction features
        features['stress_meal_interaction'] = features['stress_level'] * features['meals_skipped']
        features['mood_stress_interaction'] = features['mood_value'] * features['stress_level']

        # Rolling averages (if enough data)
        if len(df) > 7:
            features['avg_stress_7d'] = features['stress_level'].rolling(window=7, min_periods=1).mean()
            features['avg_meals_7d'] = features['total_meals'].rolling(window=7, min_periods=1).mean()
        else:
            features['avg_stress_7d'] = features['stress_level']
            features['avg_meals_7d'] = features['total_meals']

        # Fill any remaining NaN values with 0
        features = features.fillna(0)

        # Ensure all columns are numeric
        for col in features.columns:
            features[col] = self._convert_to_numeric(features[col])

        return features

    def calculate_risk_score(self, features, gender='female'):
        """
        Calculate weighted risk score based on feature categories
        """
        risk_score = 0

        # Adjust hormonal weight for females
        weights = self.feature_weights.copy()
        if gender.lower() == 'female':
            weights['hormonal'] = 0.15
            weights['stress'] = 0.18  # Slightly adjust others
        else:
            weights['hormonal'] = 0.02
            weights['stress'] = 0.23

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        weights = {k: v / total_weight for k, v in weights.items()}

        # Calculate component scores (0-1 scale)
        stress_score = min(features.get('stress_level', 0) / 10, 1.0)
        food_score = features.get('meals_skipped', 0) / 3
        activity_score = 1 - min(features.get('activity_index', 0) / 100, 1.0)
        mood_score = 1 - min(features.get('mood_value', 0) / 10, 1.0)

        # Combine with weights
        risk_score = (
                stress_score * weights['stress'] +
                food_score * weights['food_intake'] +
                activity_score * weights['activity'] +
                mood_score * weights['mood']
        )

        return min(risk_score, 1.0)

    def train(self, df, gender='female'):
        """
        Train the migraine prediction models
        """
        print("Engineering features...")
        features = self.engineer_features(df, gender)

        # Prepare target variables
        y_occurrence = self._convert_to_numeric(df.get('intensity', 0)) > 0
        y_occurrence = y_occurrence.astype(int)
        y_intensity = self._convert_to_numeric(df.get('intensity', 0))
        y_duration = self._convert_to_numeric(df.get('duration_m', 0))

        # Remove rows where we have no valid data
        valid_idx = (y_occurrence.notna()) & (features.notna().all(axis=1))

        if valid_idx.sum() == 0:
            raise ValueError("No valid training data found. Please check your CSV file.")

        X = features[valid_idx]
        y_occurrence = y_occurrence[valid_idx]
        y_intensity = y_intensity[valid_idx]
        y_duration = y_duration[valid_idx]

        print(f"Training with {len(X)} valid samples")
        print(f"Migraine occurrences: {y_occurrence.sum()} ({y_occurrence.sum() / len(y_occurrence) * 100:.1f}%)")

        # Check if we have enough positive samples
        if y_occurrence.sum() < 10:
            print("WARNING: Very few migraine occurrences in data. Model may not be reliable.")

        # Split data
        X_train, X_test, y_occ_train, y_occ_test = train_test_split(
            X, y_occurrence, test_size=0.2, random_state=42, stratify=y_occurrence if y_occurrence.sum() > 5 else None
        )
        _, _, y_int_train, y_int_test = train_test_split(
            X, y_intensity, test_size=0.2, random_state=42
        )
        _, _, y_dur_train, y_dur_test = train_test_split(
            X, y_duration, test_size=0.2, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        print("Training occurrence model...")
        self.occurrence_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            class_weight='balanced'
        )
        self.occurrence_model.fit(X_train_scaled, y_occ_train)

        print("Training intensity model...")
        self.intensity_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        # Only train on samples where migraine occurred
        migraine_idx = y_int_train > 0
        if migraine_idx.sum() > 0:
            self.intensity_model.fit(
                X_train_scaled[migraine_idx],
                y_int_train[migraine_idx]
            )
        else:
            print("WARNING: No migraine samples for intensity training")

        print("Training duration model...")
        self.duration_model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        if migraine_idx.sum() > 0:
            self.duration_model.fit(
                X_train_scaled[migraine_idx],
                y_dur_train[migraine_idx]
            )

        # Evaluate
        print("\n=== Model Evaluation ===")
        y_occ_pred = self.occurrence_model.predict(X_test_scaled)
        print("\nOccurrence Model:")
        print(classification_report(y_occ_test, y_occ_pred))

        # Feature importance
        feature_names = X.columns
        importances = self.occurrence_model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importances))

        print("\nTop 10 Important Features:")
        for feat, imp in sorted(self.feature_importance.items(),
                                key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {imp:.4f}")

        return self

    def predict(self, user_data, gender='female'):
        """
        Predict migraine occurrence, intensity, and duration
        Returns risk band, probability, and recommendations
        """
        # Engineer features
        features = self.engineer_features(user_data, gender)
        features_scaled = self.scaler.transform(features)

        # Predict occurrence probability
        occurrence_prob = self.occurrence_model.predict_proba(features_scaled)[0][1]

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

        # Predict intensity and duration if likely to occur
        predicted_intensity = 0
        predicted_duration = 0
        intensity_label = "None"

        if occurrence_prob > 0.3 and self.intensity_model is not None:
            predicted_intensity = max(0, self.intensity_model.predict(features_scaled)[0])
            predicted_duration = max(0, self.duration_model.predict(features_scaled)[0])

            if predicted_intensity < 3:
                intensity_label = "Light"
            elif predicted_intensity < 7:
                intensity_label = "Medium"
            else:
                intensity_label = "Strong"

        # Generate recommendations
        recommendations = self._generate_recommendations(
            features.iloc[0], occurrence_prob, predicted_intensity, gender
        )

        return {
            'probability': round(occurrence_prob * 100, 1),
            'risk_band': risk_band,
            'risk_label': risk_label,
            'risk_message': risk_message,
            'predicted_intensity': round(predicted_intensity, 1),
            'intensity_label': intensity_label,
            'predicted_duration_minutes': round(predicted_duration, 0),
            'recommendations': recommendations
        }

    def _generate_recommendations(self, features, probability, intensity, gender):
        """
        Generate personalized recommendations based on predictions
        """
        recommendations = []

        # Stress management
        if features.get('stress_level', 0) > 5:
            recommendations.append({
                'category': 'Stress Management',
                'action': 'Practice relaxation techniques or take a short break',
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        # Meal planning
        if features.get('meals_skipped', 0) > 0:
            recommendations.append({
                'category': 'Nutrition',
                'action': 'Try to have regular meals. Consider a light snack if meal timing is off',
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        # Sleep and rest
        if probability > 0.5:
            recommendations.append({
                'category': 'Rest',
                'action': 'Consider rescheduling non-essential activities and ensure adequate rest',
                'priority': 'high'
            })

        # Activity level
        if features.get('activity_index', 0) < 20:
            recommendations.append({
                'category': 'Activity',
                'action': 'Light physical activity may help, but avoid overexertion',
                'priority': 'low'
            })

        # Medication readiness
        if probability > 0.5 and intensity > 5:
            recommendations.append({
                'category': 'Medication',
                'action': 'Have your migraine medication ready and consider taking it early',
                'priority': 'high'
            })

        # Hydration
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
            'intensity_model': self.intensity_model,
            'duration_model': self.duration_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'feature_weights': self.feature_weights
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    def load_model(self, filepath='migraine_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.occurrence_model = model_data['occurrence_model']
        self.intensity_model = model_data['intensity_model']
        self.duration_model = model_data['duration_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.feature_weights = model_data['feature_weights']
        print(f"Model loaded from {filepath}")
        return self


# Training Pipeline
class ModelTrainer:
    """
    Handles the complete training pipeline for migraine prediction
    """

    @staticmethod
    def load_csv_data(filepath):
        """Load and preprocess CSV data"""
        print(f"Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"Loaded {len(df)} records")
        print(f"Columns: {list(df.columns)[:20]}...")  # Show first 20 columns
        return df

    @staticmethod
    def validate_data(df):
        """Validate data quality and show statistics"""
        print("\n=== Data Validation ===")
        print(f"Total records: {len(df)}")

        if 'time' in df.columns:
            try:
                dates = pd.to_datetime(df['time'], unit='ms', errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    print(f"Date range: {valid_dates.min()} to {valid_dates.max()}")
            except Exception as e:
                print(f"Could not parse dates: {e}")

        # Check for migraine records
        if 'intensity' in df.columns:
            intensity_numeric = pd.to_numeric(df['intensity'], errors='coerce')
            migraine_count = (intensity_numeric > 0).sum()
            total_valid = intensity_numeric.notna().sum()
            if total_valid > 0:
                print(f"Migraine records: {migraine_count} ({migraine_count / total_valid * 100:.1f}%)")
                print(f"Intensity distribution:")
                print(intensity_numeric[intensity_numeric > 0].value_counts().sort_index())

        # Check data completeness
        print(f"\nMissing data percentages (top 10):")
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        for col, pct in missing.head(10).items():
            if pct > 0:
                print(f"  {col}: {pct:.1f}%")

        return True

    @staticmethod
    def train_model_from_csv(csv_filepath, gender='female', save_path='migraine_model.pkl'):
        """
        Complete training pipeline from CSV file
        """
        # Load data
        df = ModelTrainer.load_csv_data(csv_filepath)

        # Validate
        ModelTrainer.validate_data(df)

        # Initialize and train model
        print("\n=== Starting Model Training ===")
        model = MigrainePredictionModel()
        model.train(df, gender=gender)

        # Save model
        model.save_model(save_path)

        return model


# Example usage
if __name__ == "__main__":
    print("=== Migraine Prediction ML Model - Training Module ===\n")

    # Train with your CSV file
    model = ModelTrainer.train_model_from_csv(
        csv_filepath='event_dump.csv',
        gender='female',
        save_path='migraine_model.pkl'
    )

    print("\n" + "=" * 60)
    print("Training complete! Model saved and ready for use.")
    print("=" * 60)