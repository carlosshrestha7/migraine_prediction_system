import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, mean_squared_error, accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')


class MigrainePredictionModel:
    """
    ML Model for predicting migraine occurrence, intensity, and duration
    Updated to handle health_data CSV structure
    """

    def __init__(self):
        self.occurrence_model = None
        self.intensity_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

        # Feature weights by category
        self.feature_weights = {
            'sleep': 0.25,
            'stress': 0.20,
            'food_intake': 0.15,
            'weather': 0.10,
            'hormonal': 0.15,
            'mood': 0.10,
            'activity': 0.05
        }

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
        Create engineered features from health_data CSV structure
        """
        features = pd.DataFrame()

        # Time-based features
        if 'timestamp' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
            features['hour_of_day'] = df['datetime'].dt.hour.fillna(0)
            features['day_of_week'] = df['datetime'].dt.dayofweek.fillna(0)
            features['day_of_month'] = df['datetime'].dt.day.fillna(0)
            features['month'] = df['datetime'].dt.month.fillna(0)
        elif 'timestamp_dt' in df.columns:
            df['datetime'] = pd.to_datetime(df['timestamp_dt'], errors='coerce')
            features['hour_of_day'] = df['datetime'].dt.hour.fillna(0)
            features['day_of_week'] = df['datetime'].dt.dayofweek.fillna(0)
            features['day_of_month'] = df['datetime'].dt.day.fillna(0)
            features['month'] = df['datetime'].dt.month.fillna(0)
        else:
            features['hour_of_day'] = 0
            features['day_of_week'] = 0
            features['day_of_month'] = 0
            features['month'] = 0

        # Core health features from your dataset
        features['stress_intensity'] = self._get_column(df, 'stress_intensity', 'stress')
        features['sleep_duration'] = self._get_column(df, 'sleep_duration')
        features['sleep_deficit'] = self._get_column(df, 'sleep_deficit')
        features['missed_meal'] = self._get_column(df, 'missed_meal')
        features['menstruation'] = self._get_column(df, 'menstruation')
        features['delivery'] = self._get_column(df, 'delivery')
        features['migraine_days_per_month'] = self._get_column(df, 'migraine_days_per_month')

        # Probability features (if available)
        features['p_stress'] = self._get_column(df, 'p_stress')
        features['p_hormones'] = self._get_column(df, 'p_hormones')
        features['p_sleep'] = self._get_column(df, 'p_sleep')
        features['p_weather'] = self._get_column(df, 'p_weather')
        features['p_meals'] = self._get_column(df, 'p_meals')
        features['migraine_probability'] = self._get_column(df, 'migraine_probability')

        # Derived features
        features['sleep_quality'] = 1 / (1 + features['sleep_deficit'])  # Inverse of deficit
        features['is_well_rested'] = (features['sleep_duration'] >= 7).astype(int)
        features['high_stress'] = (features['stress_intensity'] > 7).astype(int)
        features['hormonal_event'] = (features['menstruation'] | features['delivery']).astype(int)

        # Interaction features
        features['stress_sleep_interaction'] = features['stress_intensity'] * features['sleep_deficit']
        features['stress_meal_interaction'] = features['stress_intensity'] * features['missed_meal']
        features['hormone_stress_interaction'] = features['hormonal_event'] * features['stress_intensity']

        # Rolling averages (if enough data per person)
        if 'person_id' in df.columns and len(df) > 100:
            # Group by person for rolling calculations
            for person_id in df['person_id'].unique()[:100]:  # Limit for performance
                person_mask = df['person_id'] == person_id
                person_indices = df[person_mask].index

                if len(person_indices) > 7:
                    features.loc[person_indices, 'avg_stress_7d'] = (
                        features.loc[person_indices, 'stress_intensity']
                        .rolling(window=7, min_periods=1).mean()
                    )
                    features.loc[person_indices, 'avg_sleep_7d'] = (
                        features.loc[person_indices, 'sleep_duration']
                        .rolling(window=7, min_periods=1).mean()
                    )

        # Fill rolling averages if not calculated
        if 'avg_stress_7d' not in features.columns:
            features['avg_stress_7d'] = features['stress_intensity']
        if 'avg_sleep_7d' not in features.columns:
            features['avg_sleep_7d'] = features['sleep_duration']

        # Previous day features (shifted by person if possible)
        if 'person_id' in df.columns:
            features['prev_stress'] = df.groupby('person_id')['stress_intensity'].shift(1).fillna(0)
            if 'migraine' in df.columns:
                features['prev_migraine'] = df.groupby('person_id')['migraine'].shift(1).fillna(0)
        else:
            features['prev_stress'] = features['stress_intensity'].shift(1).fillna(0)

        # Gender-specific features
        features['gender_female'] = 1 if gender.lower() == 'female' else 0

        # Fill any remaining NaN values
        features = features.fillna(0)

        # Ensure all columns are numeric
        for col in features.columns:
            features[col] = self._convert_to_numeric(features[col])

        return features

    def train(self, df, gender='female', sample_size=None):
        """Train the migraine prediction models"""

        # Sample data if too large
        if sample_size and len(df) > sample_size:
            print(f"📊 Sampling {sample_size} records from {len(df)} total records...")
            df = df.sample(n=sample_size, random_state=42)

        print("Engineering features...")
        features = self.engineer_features(df, gender)

        # Prepare target variable - use 'migraine' column
        if 'migraine' not in df.columns:
            raise ValueError("'migraine' column not found in dataset!")

        y_occurrence = self._convert_to_numeric(df['migraine'], 0)
        y_occurrence = (y_occurrence > 0).astype(int)

        # Remove invalid rows
        valid_idx = (y_occurrence.notna()) & (features.notna().all(axis=1))

        if valid_idx.sum() == 0:
            raise ValueError("No valid training data found. Please check your CSV file.")

        X = features[valid_idx]
        y_occurrence = y_occurrence[valid_idx]

        print(f"\n✅ Training with {len(X)} valid samples")
        print(f"📊 Features used: {len(X.columns)}")
        print(f"🔴 Migraine occurrences: {y_occurrence.sum():,} ({y_occurrence.sum() / len(y_occurrence) * 100:.2f}%)")
        print(
            f"🟢 Non-migraine days: {(~y_occurrence.astype(bool)).sum():,} ({(~y_occurrence.astype(bool)).sum() / len(y_occurrence) * 100:.2f}%)")

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
        X_scaled = self.scaler.transform(features)

        # Predict occurrence probability
        occurrence_prob = self.occurrence_model.predict_proba(X_scaled)[0][1]

        # Calculate risk band
        if occurrence_prob < 0.20:
            risk_band = "green"
            risk_label = "Low Risk"
            risk_message = "Low probability of migraine today"
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

    def _identify_risk_factors(self, features):
        """Identify current risk factors"""
        risk_factors = []

        if features.get('stress_intensity', 0) > 7:
            risk_factors.append("High stress level")
        if features.get('sleep_deficit', 0) > 2:
            risk_factors.append("Sleep deficit")
        if features.get('missed_meal', 0) > 0:
            risk_factors.append("Missed meals")
        if features.get('hormonal_event', 0) > 0:
            risk_factors.append("Hormonal changes")
        if features.get('sleep_duration', 0) < 6:
            risk_factors.append("Insufficient sleep")

        return risk_factors

    def _generate_recommendations(self, features, probability, gender):
        """Generate personalized recommendations"""
        recommendations = []

        # Stress management
        if features.get('stress_intensity', 0) > 7:
            recommendations.append({
                'category': 'Stress Management',
                'action': 'Your stress level is high. Practice deep breathing or meditation',
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        # Sleep recommendations
        if features.get('sleep_deficit', 0) > 1:
            recommendations.append({
                'category': 'Sleep',
                'action': f"You have a sleep deficit. Try to rest early tonight",
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        # Meal planning
        if features.get('missed_meal', 0) > 0:
            recommendations.append({
                'category': 'Nutrition',
                'action': 'Avoid skipping meals. Have regular, balanced meals',
                'priority': 'high'
            })

        # Preventive measures for high risk
        if probability > 0.6:
            recommendations.append({
                'category': 'Prevention',
                'action': 'Consider taking preventive medication if prescribed',
                'priority': 'high'
            })

        # Hydration
        recommendations.append({
            'category': 'Hydration',
            'action': 'Ensure you stay well hydrated throughout the day',
            'priority': 'medium' if probability > 0.4 else 'low'
        })

        # Activity
        if features.get('stress_intensity', 0) > 5:
            recommendations.append({
                'category': 'Activity',
                'action': 'Light exercise or a short walk may help reduce stress',
                'priority': 'low'
            })

        return recommendations

    def save_model(self, filepath='migraine_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'occurrence_model': self.occurrence_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'feature_weights': self.feature_weights
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")

    def load_model(self, filepath='migraine_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.occurrence_model = model_data['occurrence_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.feature_weights = model_data['feature_weights']
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
        print(f"📋 Columns found: {list(df.columns)}")
        return df

    @staticmethod
    def validate_data(df):
        """Validate data quality and show statistics"""
        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)
        print(f"Total records: {len(df):,}")

        # Check for required columns
        required_cols = ['migraine', 'stress_intensity', 'sleep_duration']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing required columns: {missing_cols}")

        # Date range
        if 'timestamp' in df.columns:
            try:
                dates = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    print(f"📅 Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
                    print(f"   Duration: {(valid_dates.max() - valid_dates.min()).days} days")
            except Exception as e:
                print(f"⚠️  Could not parse dates: {e}")

        # Migraine statistics
        if 'migraine' in df.columns:
            migraine_numeric = pd.to_numeric(df['migraine'], errors='coerce')
            migraine_count = (migraine_numeric > 0).sum()
            total_valid = migraine_numeric.notna().sum()
            if total_valid > 0:
                print(f"\n🔴 Migraine Events: {migraine_count:,} ({migraine_count / total_valid * 100:.2f}%)")
                print(
                    f"🟢 Healthy Days: {total_valid - migraine_count:,} ({(total_valid - migraine_count) / total_valid * 100:.2f}%)")

        # Stress statistics
        if 'stress_intensity' in df.columns:
            stress = pd.to_numeric(df['stress_intensity'], errors='coerce')
            print(f"\n😰 Stress Statistics:")
            print(f"   Mean: {stress.mean():.2f}")
            print(f"   High stress days (>7): {(stress > 7).sum():,}")

        # Sleep statistics
        if 'sleep_duration' in df.columns:
            sleep = pd.to_numeric(df['sleep_duration'], errors='coerce')
            print(f"\n😴 Sleep Statistics:")
            print(f"   Mean duration: {sleep.mean():.2f} hours")
            print(f"   Insufficient sleep (<6h): {(sleep < 6).sum():,}")

        # Data completeness
        print(f"\n📉 Missing Data:")
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        has_missing = False
        for col, pct in missing.items():
            if pct > 0:
                has_missing = True
                print(f"   {col}: {pct:.1f}%")
        if not has_missing:
            print("   ✅ No missing data!")

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
    # Use sample_size to limit data for faster training (optional)
    model = ModelTrainer.train_model_from_csv(
        csv_filepath='synthetic_data_10_000/health_data_10000_365.csv',
        gender='female',
        save_path='migraine_model.pkl',
        sample_size=100000  # Use 100k samples for faster training, or None for all data
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("📦 Model saved and ready for predictions.")
    print("💡 Use model.predict(new_data) to make predictions.")
    print("=" * 60)