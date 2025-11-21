import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import warnings
from pathlib import Path

warnings.filterwarnings('ignore')


class MigrainePredictionModel:
    """
    ML Model for predicting migraine occurrence with multi-source data support
    Supports health_data.csv and weather_data.csv integration
    """

    def __init__(self):
        self.occurrence_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

        # Feature weights by category
        self.feature_weights = {
            'sleep': 0.20,
            'stress': 0.20,
            'food_intake': 0.12,
            'weather': 0.18,  # Increased weight for weather
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
        Create engineered features from health_data with optional weather data
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

        # Core health features
        features['stress_intensity'] = self._get_column(df, 'stress_intensity', 'stress')
        features['sleep_duration'] = self._get_column(df, 'sleep_duration')
        features['sleep_deficit'] = self._get_column(df, 'sleep_deficit')
        features['missed_meal'] = self._get_column(df, 'missed_meal')
        features['menstruation'] = self._get_column(df, 'menstruation')
        features['delivery'] = self._get_column(df, 'delivery')
        features['migraine_days_per_month'] = self._get_column(df, 'migraine_days_per_month')

        # Probability features
        features['p_stress'] = self._get_column(df, 'p_stress')
        features['p_hormones'] = self._get_column(df, 'p_hormones')
        features['p_sleep'] = self._get_column(df, 'p_sleep')
        features['p_weather'] = self._get_column(df, 'p_weather')
        features['p_meals'] = self._get_column(df, 'p_meals')
        features['migraine_probability'] = self._get_column(df, 'migraine_probability')

        # === WEATHER FEATURES (if available) ===
        weather_features = [
            'temp_min', 'temp_max', 'temp_mean', 'temp_quick_change',
            'wind_min', 'wind_max', 'wind_mean', 'wind_quick_change',
            'pressure_min', 'pressure_max', 'pressure_mean', 'pressure_quick_change',
            'sun_irr_min', 'sun_irr_max', 'sun_irr_mean',
            'sun_time_min', 'sun_time_max', 'sun_time_mean',
            'sun_irr_quick_change', 'sun_time_quick_change',
            'precip_min', 'precip_max', 'precip_mean', 'precip_total',
            'precipitation_quick_change',
            'cloud_min', 'cloud_max', 'cloud_mean', 'cloud_quick_change'
        ]

        for weather_col in weather_features:
            features[weather_col] = self._get_column(df, weather_col)

        # === WEATHER-DERIVED FEATURES ===
        # Temperature features
        features['temp_range'] = features['temp_max'] - features['temp_min']
        features['is_extreme_temp'] = ((features['temp_mean'] < 5) |
                                       (features['temp_mean'] > 30)).astype(int)
        features['is_temp_unstable'] = (features['temp_quick_change'] > 5).astype(int)

        # Pressure features (barometric pressure changes are strong migraine triggers)
        features['pressure_range'] = features['pressure_max'] - features['pressure_min']
        features['is_pressure_drop'] = (features['pressure_quick_change'] < -2).astype(int)
        features['is_low_pressure'] = (features['pressure_mean'] < 1010).astype(int)

        # Wind features
        features['wind_range'] = features['wind_max'] - features['wind_min']
        features['is_windy'] = (features['wind_mean'] > 20).astype(int)

        # Precipitation and humidity indicators
        features['is_rainy'] = (features['precip_total'] > 5).astype(int)
        features['is_cloudy'] = (features['cloud_mean'] > 70).astype(int)

        # Sun/brightness features
        features['sun_irr_range'] = features['sun_irr_max'] - features['sun_irr_min']
        features['is_bright'] = (features['sun_irr_mean'] > 600).astype(int)
        features['sun_time_range'] = features['sun_time_max'] - features['sun_time_min']

        # === HEALTH-DERIVED FEATURES ===
        features['sleep_quality'] = 1 / (1 + features['sleep_deficit'])
        features['is_well_rested'] = (features['sleep_duration'] >= 7).astype(int)
        features['high_stress'] = (features['stress_intensity'] > 7).astype(int)
        features['hormonal_event'] = (features['menstruation'] | features['delivery']).astype(int)

        # === INTERACTION FEATURES ===
        # Health interactions
        features['stress_sleep_interaction'] = features['stress_intensity'] * features['sleep_deficit']
        features['stress_meal_interaction'] = features['stress_intensity'] * features['missed_meal']
        features['hormone_stress_interaction'] = features['hormonal_event'] * features['stress_intensity']

        # Weather-health interactions (key for finding relationships!)
        features['pressure_stress_interaction'] = features['pressure_quick_change'] * features['stress_intensity']
        features['temp_change_sleep_interaction'] = features['temp_quick_change'] * features['sleep_deficit']
        features['weather_hormone_interaction'] = features['pressure_quick_change'] * features['hormonal_event']
        features['cloudy_stress_interaction'] = features['is_cloudy'] * features['high_stress']
        features['pressure_sleep_interaction'] = features['is_pressure_drop'] * features['sleep_deficit']

        # === ROLLING AVERAGES ===
        if 'person_id' in df.columns and len(df) > 100:
            for person_id in df['person_id'].unique()[:100]:
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
                    features.loc[person_indices, 'avg_pressure_7d'] = (
                        features.loc[person_indices, 'pressure_mean']
                        .rolling(window=7, min_periods=1).mean()
                    )

        # Fill rolling averages if not calculated
        if 'avg_stress_7d' not in features.columns:
            features['avg_stress_7d'] = features['stress_intensity']
        if 'avg_sleep_7d' not in features.columns:
            features['avg_sleep_7d'] = features['sleep_duration']
        if 'avg_pressure_7d' not in features.columns:
            features['avg_pressure_7d'] = features['pressure_mean']

        # === PREVIOUS DAY FEATURES ===
        if 'person_id' in df.columns:
            features['prev_stress'] = df.groupby('person_id')['stress_intensity'].shift(1).fillna(0)
            features['prev_pressure'] = df.groupby('person_id')['pressure_mean'].shift(1).fillna(0)
            if 'migraine' in df.columns:
                features['prev_migraine'] = df.groupby('person_id')['migraine'].shift(1).fillna(0)
        else:
            features['prev_stress'] = features['stress_intensity'].shift(1).fillna(0)
            features['prev_pressure'] = features['pressure_mean'].shift(1).fillna(0)

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

        print("🔧 Engineering features...")
        features = self.engineer_features(df, gender)

        # Check which weather features are available
        weather_cols = [col for col in features.columns if any(
            w in col for w in ['temp', 'pressure', 'wind', 'sun', 'precip', 'cloud', 'weather']
        )]
        if weather_cols:
            print(f"🌤️  Weather features found: {len(weather_cols)}")
        else:
            print("⚠️  No weather data detected - training without weather features")

        # Prepare target variable
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

        if y_test.sum() > 0:
            print(f"📊 ROC-AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Feature importance
        feature_names = X.columns
        importances = self.occurrence_model.feature_importances_
        self.feature_importance = dict(zip(feature_names, importances))

        print("\n🏆 Top 20 Most Important Features:")
        for i, (feat, imp) in enumerate(sorted(self.feature_importance.items(),
                                               key=lambda x: x[1], reverse=True)[:20], 1):
            bar = "█" * int(imp * 100)
            # Highlight weather-related features
            emoji = "🌤️ " if any(
                w in feat for w in ['temp', 'pressure', 'wind', 'sun', 'precip', 'cloud', 'weather']) else "   "
            print(f"{emoji}{i:2d}. {feat:35s} {imp:6.4f} {bar}")

        # Weather feature importance summary
        weather_importance = {k: v for k, v in self.feature_importance.items()
                              if any(w in k for w in ['temp', 'pressure', 'wind', 'sun', 'precip', 'cloud', 'weather'])}
        if weather_importance:
            total_weather_imp = sum(weather_importance.values())
            print(f"\n🌤️  Total Weather Feature Importance: {total_weather_imp:.4f} ({total_weather_imp * 100:.1f}%)")
            print("    Top weather factors:")
            for feat, imp in sorted(weather_importance.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"    • {feat}: {imp:.4f}")

        print("=" * 60)
        return self

    def predict(self, user_data, gender='female'):
        """Predict migraine occurrence probability"""
        features = self.engineer_features(user_data, gender)
        X_scaled = self.scaler.transform(features)

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

        recommendations = self._generate_recommendations(features.iloc[0], occurrence_prob, gender)

        return {
            'probability': round(occurrence_prob * 100, 1),
            'risk_band': risk_band,
            'risk_label': risk_label,
            'risk_message': risk_message,
            'recommendations': recommendations,
            'top_risk_factors': self._identify_risk_factors(features.iloc[0])
        }

    def _identify_risk_factors(self, features):
        """Identify current risk factors including weather"""
        risk_factors = []

        # Health factors
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

        # Weather factors
        if features.get('is_pressure_drop', 0) > 0:
            risk_factors.append("Barometric pressure drop")
        if features.get('temp_quick_change', 0) > 5:
            risk_factors.append("Rapid temperature change")
        if features.get('is_windy', 0) > 0:
            risk_factors.append("High winds")
        if features.get('is_bright', 0) > 0:
            risk_factors.append("Bright sunlight")

        return risk_factors

    def _generate_recommendations(self, features, probability, gender):
        """Generate personalized recommendations including weather-based ones"""
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

        # Weather-based recommendations
        if features.get('is_pressure_drop', 0) > 0:
            recommendations.append({
                'category': 'Weather Alert',
                'action': 'Barometric pressure is dropping - stay hydrated and consider preventive measures',
                'priority': 'high'
            })

        if features.get('is_bright', 0) > 0:
            recommendations.append({
                'category': 'Weather',
                'action': 'Bright sunlight expected - wear sunglasses and avoid prolonged exposure',
                'priority': 'medium'
            })

        if features.get('temp_quick_change', 0) > 5:
            recommendations.append({
                'category': 'Weather',
                'action': 'Rapid temperature changes - dress in layers and stay comfortable',
                'priority': 'medium'
            })

        # Preventive measures
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
    """Handles the complete training pipeline with multi-source data"""

    @staticmethod
    def load_csv_data(filepath):
        """Load and preprocess CSV data"""
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
        print(f"📋 Columns found: {list(df.columns)[:10]}{'...' if len(df.columns) > 10 else ''}")
        return df

    @staticmethod
    def merge_weather_data(health_df, weather_filepath):
        """
        Merge weather data with health data based on date
        """
        print(f"\n🌤️  Loading weather data from {weather_filepath}...")
        weather_df = pd.read_csv(weather_filepath)
        print(f"✅ Loaded {len(weather_df):,} weather records")

        # Convert health data timestamp to date
        if 'timestamp' in health_df.columns:
            health_df['date'] = pd.to_datetime(health_df['timestamp'], unit='s', errors='coerce').dt.date
        elif 'timestamp_dt' in health_df.columns:
            health_df['date'] = pd.to_datetime(health_df['timestamp_dt'], errors='coerce').dt.date
        else:
            print("⚠️  WARNING: No timestamp column found in health data")
            return health_df

        # Convert weather date to date object
        weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce').dt.date

        # Count records before merge
        before_count = len(health_df)

        # Merge on date
        merged_df = health_df.merge(weather_df, on='date', how='left')

        after_count = len(merged_df)
        matched_weather = merged_df[weather_df.columns[1]].notna().sum()

        print(f"🔗 Merge complete:")
        print(f"   • Health records: {before_count:,}")
        print(f"   • Records with weather data: {matched_weather:,} ({matched_weather / before_count * 100:.1f}%)")
        print(f"   • Total columns: {len(merged_df.columns)}")

        if matched_weather == 0:
            print("⚠️  WARNING: No matching dates found between health and weather data!")
            print("   Check that date formats are compatible")

        return merged_df

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

        # Check for weather columns
        weather_cols = [col for col in df.columns if any(
            w in col for w in ['temp', 'pressure', 'wind', 'sun', 'precip', 'cloud']
        )]
        if weather_cols:
            print(f"🌤️  Weather columns found: {len(weather_cols)}")
        else:
            print("⚠️  No weather columns detected")

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

        # Weather statistics (if available)
        if 'pressure_mean' in df.columns:
            pressure = pd.to_numeric(df['pressure_mean'], errors='coerce')
            print(f"\n🌡️  Pressure Statistics:")
            print(f"   Mean: {pressure.mean():.1f} hPa")
            print(f"   Range: {pressure.min():.1f} - {pressure.max():.1f} hPa")

        if 'temp_mean' in df.columns:
            temp = pd.to_numeric(df['temp_mean'], errors='coerce')
            print(f"\n🌡️  Temperature Statistics:")
            print(f"   Mean: {temp.mean():.1f}°C")
            print(f"   Range: {temp.min():.1f} - {temp.max():.1f}°C")

        print("=" * 60)
        return True

    @staticmethod
    def train_model_from_csv(health_csv, weather_csv=None, gender='female',
                             save_path='migraine_model.pkl', sample_size=None):
        """
        Complete training pipeline with optional weather data

        Args:
            health_csv: Path to health_data CSV file
            weather_csv: Optional path to weather_data CSV file
            gender: Gender for training ('female' or 'male')
            save_path: Where to save the trained model
            sample_size: Optional limit on training samples
        """
        # Load health data
        df = ModelTrainer.load_csv_data(health_csv)

        # Merge weather data if provided
        if weather_csv:
            if Path(weather_csv).exists():
                df = ModelTrainer.merge_weather_data(df, weather_csv)
            else:
                print(f"⚠️  Weather file not found: {weather_csv}")
                print("   Training without weather data...")

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
    print("🧠 MIGRAINE PREDICTION ML MODEL - WITH WEATHER INTEGRATION")
    print("=" * 60 + "\n")

    # Train with health + weather data
    model = ModelTrainer.train_model_from_csv(
        health_csv='synthetic_data_10_000/health_data_10000_365.csv',
        weather_csv='synthetic_data_10_000/weather_data.csv',  # Add your weather CSV here!
        gender='female',
        save_path='migraine_model_weather.pkl',
        sample_size=100000  # Use None for all data
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("📦 Model saved with weather features integrated.")
    print("💡 The model can now detect relationships between weather and migraines!")
    print("=" * 60)