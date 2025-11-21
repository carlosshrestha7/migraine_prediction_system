import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import warnings

warnings.filterwarnings('ignore')


class MigrainePredictionModel:
    """
    ML Model for predicting migraine occurrence with weather data integration
    """

    def __init__(self):
        self.occurrence_model = None
        self.scaler = StandardScaler()
        self.feature_importance = {}

        # Feature weights by category
        self.feature_weights = {
            'sleep': 0.20,
            'stress': 0.20,
            'food_intake': 0.10,
            'weather': 0.20,
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

    def merge_weather_data(self, health_df, weather_df):
        """
        Merge health data with weather data based on date
        """
        print("🌦️  Merging weather data with health data...")

        # Prepare health data dates
        if 'timestamp' in health_df.columns:
            health_df['date'] = pd.to_datetime(health_df['timestamp'], unit='s', errors='coerce').dt.date
        elif 'timestamp_dt' in health_df.columns:
            health_df['date'] = pd.to_datetime(health_df['timestamp_dt'], errors='coerce').dt.date
        else:
            raise ValueError("Health data must have 'timestamp' or 'timestamp_dt' column")

        # Prepare weather data dates
        weather_df['date'] = pd.to_datetime(weather_df['date'], errors='coerce').dt.date

        # Merge on date
        merged_df = health_df.merge(weather_df, on='date', how='left', suffixes=('', '_weather'))

        print(f"✅ Merged {len(merged_df):,} records")
        weather_coverage = merged_df['temp_mean'].notna().sum() if 'temp_mean' in merged_df.columns else 0
        print(
            f"📊 Weather data available for {weather_coverage:,} records ({weather_coverage / len(merged_df) * 100:.1f}%)")

        return merged_df

    def engineer_features(self, df, gender='female'):
        """
        Create engineered features including weather data
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

        # ==================== WEATHER FEATURES ====================

        # Temperature features
        features['temp_mean'] = self._get_column(df, 'temp_mean')
        features['temp_min'] = self._get_column(df, 'temp_min')
        features['temp_max'] = self._get_column(df, 'temp_max')
        features['temp_range'] = features['temp_max'] - features['temp_min']
        features['temp_quick_change'] = self._get_column(df, 'temp_quick_change').astype(int)

        # Pressure features (major migraine trigger)
        features['pressure_mean'] = self._get_column(df, 'pressure_mean')
        features['pressure_min'] = self._get_column(df, 'pressure_min')
        features['pressure_max'] = self._get_column(df, 'pressure_max')
        features['pressure_range'] = features['pressure_max'] - features['pressure_min']
        features['pressure_quick_change'] = self._get_column(df, 'pressure_quick_change').astype(int)

        # Wind features
        features['wind_mean'] = self._get_column(df, 'wind_mean')
        features['wind_max'] = self._get_column(df, 'wind_max')
        features['wind_quick_change'] = self._get_column(df, 'wind_quick_change').astype(int)

        # Precipitation features
        features['precip_mean'] = self._get_column(df, 'precip_mean')
        features['precip_total'] = self._get_column(df, 'precip_total')
        features['precipitation_quick_change'] = self._get_column(df, 'precipitation_quick_change').astype(int)

        # Cloud cover features
        features['cloud_mean'] = self._get_column(df, 'cloud_mean')
        features['cloud_quick_change'] = self._get_column(df, 'cloud_quick_change').astype(int)

        # Sun/light features
        features['sun_irr_mean'] = self._get_column(df, 'sun_irr_mean')
        features['sun_time_mean'] = self._get_column(df, 'sun_time_mean')
        features['sun_irr_quick_change'] = self._get_column(df, 'sun_irr_quick_change').astype(int)

        # Weather change aggregation (total instability)
        features['total_weather_changes'] = (
                features['temp_quick_change'] +
                features['pressure_quick_change'] +
                features['wind_quick_change'] +
                features['precipitation_quick_change'] +
                features['cloud_quick_change'] +
                features['sun_irr_quick_change']
        )

        # Weather severity indicators
        features['extreme_temp'] = ((features['temp_mean'] < 0) | (features['temp_mean'] > 30)).astype(int)
        features['high_wind'] = (features['wind_max'] > 10).astype(int)
        features['stormy_weather'] = ((features['precip_total'] > 5) | (features['wind_max'] > 15)).astype(int)

        # ==================== DERIVED FEATURES ====================

        # Sleep quality
        features['sleep_quality'] = 1 / (1 + features['sleep_deficit'])
        features['is_well_rested'] = (features['sleep_duration'] >= 7).astype(int)
        features['high_stress'] = (features['stress_intensity'] > 7).astype(int)
        features['hormonal_event'] = (features['menstruation'] | features['delivery']).astype(int)

        # Probability features (if available)
        features['p_stress'] = self._get_column(df, 'p_stress')
        features['p_hormones'] = self._get_column(df, 'p_hormones')
        features['p_sleep'] = self._get_column(df, 'p_sleep')
        features['p_weather'] = self._get_column(df, 'p_weather')
        features['p_meals'] = self._get_column(df, 'p_meals')
        features['migraine_probability'] = self._get_column(df, 'migraine_probability')

        # ==================== INTERACTION FEATURES ====================

        # Health interactions
        features['stress_sleep_interaction'] = features['stress_intensity'] * features['sleep_deficit']
        features['stress_meal_interaction'] = features['stress_intensity'] * features['missed_meal']
        features['hormone_stress_interaction'] = features['hormonal_event'] * features['stress_intensity']

        # Weather-health interactions (KEY FOR MIGRAINE PREDICTION)
        features['pressure_stress_interaction'] = features['pressure_quick_change'] * features['stress_intensity']
        features['weather_sleep_interaction'] = features['total_weather_changes'] * features['sleep_deficit']
        features['pressure_hormone_interaction'] = features['pressure_quick_change'] * features['hormonal_event']
        features['temp_change_stress'] = features['temp_quick_change'] * features['stress_intensity']

        # Combined risk score
        features['combined_risk_score'] = (
                features['stress_intensity'] * 0.25 +
                features['sleep_deficit'] * 0.20 +
                features['total_weather_changes'] * 0.20 +
                features['pressure_quick_change'] * 3.0 +
                features['hormonal_event'] * 2.0
        )

        # ==================== ROLLING/LAGGED FEATURES ====================

        # Rolling averages for weather (helps detect trends)
        if len(df) > 7:
            features['pressure_7d_avg'] = features['pressure_mean'].rolling(window=7, min_periods=1).mean()
            features['temp_7d_avg'] = features['temp_mean'].rolling(window=7, min_periods=1).mean()
            features['pressure_deviation'] = features['pressure_mean'] - features['pressure_7d_avg']
            features['temp_deviation'] = features['temp_mean'] - features['temp_7d_avg']

            # Health rolling averages
            features['avg_stress_7d'] = features['stress_intensity'].rolling(window=7, min_periods=1).mean()
            features['avg_sleep_7d'] = features['sleep_duration'].rolling(window=7, min_periods=1).mean()
        else:
            features['pressure_7d_avg'] = features['pressure_mean']
            features['temp_7d_avg'] = features['temp_mean']
            features['pressure_deviation'] = 0
            features['temp_deviation'] = 0
            features['avg_stress_7d'] = features['stress_intensity']
            features['avg_sleep_7d'] = features['sleep_duration']

        # Previous day features
        features['prev_pressure'] = features['pressure_mean'].shift(1).fillna(features['pressure_mean'])
        features['prev_temp'] = features['temp_mean'].shift(1).fillna(features['temp_mean'])
        features['pressure_change_1d'] = features['pressure_mean'] - features['prev_pressure']
        features['temp_change_1d'] = features['temp_mean'] - features['prev_temp']
        features['prev_stress'] = features['stress_intensity'].shift(1).fillna(0)

        if 'migraine' in df.columns:
            features['prev_migraine'] = self._convert_to_numeric(df['migraine']).shift(1).fillna(0)

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
            n_estimators=150,
            max_depth=20,
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
            emoji = "🌦️ " if any(
                w in feat for w in ['weather', 'pressure', 'temp', 'wind', 'precip', 'cloud', 'sun']) else ""
            print(f"  {i:2d}. {emoji}{feat:35s} {imp:6.4f} {bar}")

        # Weather-specific feature importance
        weather_features = {k: v for k, v in self.feature_importance.items()
                            if any(w in k for w in ['pressure', 'temp', 'wind', 'precip', 'cloud', 'sun', 'weather'])}

        if weather_features:
            print("\n🌦️  Weather Feature Importance:")
            total_weather_importance = sum(weather_features.values())
            print(
                f"   Total weather contribution: {total_weather_importance:.4f} ({total_weather_importance * 100:.1f}%)")
            for feat, imp in sorted(weather_features.items(), key=lambda x: x[1], reverse=True)[:10]:
                print(f"   • {feat:35s} {imp:6.4f}")

        print("=" * 60)
        return self

    def predict(self, user_data, gender='female'):
        """
        Predict migraine occurrence probability
        """
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
        """Identify current risk factors including weather"""
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

        # Weather factors
        if features.get('pressure_quick_change', 0) > 0:
            risk_factors.append("Barometric pressure changes")
        if features.get('temp_quick_change', 0) > 0:
            risk_factors.append("Temperature fluctuations")
        if features.get('total_weather_changes', 0) > 2:
            risk_factors.append("Unstable weather conditions")
        if features.get('stormy_weather', 0) > 0:
            risk_factors.append("Stormy weather")

        return risk_factors

    def _generate_recommendations(self, features, probability, gender):
        """Generate personalized recommendations including weather-based advice"""
        recommendations = []

        if features.get('stress_intensity', 0) > 7:
            recommendations.append({
                'category': 'Stress Management',
                'action': 'Your stress level is high. Practice deep breathing or meditation',
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        if features.get('sleep_deficit', 0) > 1:
            recommendations.append({
                'category': 'Sleep',
                'action': f"You have a sleep deficit. Try to rest early tonight",
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        if features.get('missed_meal', 0) > 0:
            recommendations.append({
                'category': 'Nutrition',
                'action': 'Avoid skipping meals. Have regular, balanced meals',
                'priority': 'high'
            })

        # Weather-specific recommendations
        if features.get('pressure_quick_change', 0) > 0:
            recommendations.append({
                'category': 'Weather Alert',
                'action': 'Barometric pressure is changing. Stay hydrated and avoid bright lights',
                'priority': 'high' if probability > 0.6 else 'medium'
            })

        if features.get('total_weather_changes', 0) > 2:
            recommendations.append({
                'category': 'Weather Alert',
                'action': 'Unstable weather detected. Consider staying indoors and maintaining routine',
                'priority': 'medium'
            })

        if features.get('stormy_weather', 0) > 0:
            recommendations.append({
                'category': 'Weather Alert',
                'action': 'Stormy conditions may trigger migraines. Have medication ready',
                'priority': 'high' if probability > 0.5 else 'medium'
            })

        if probability > 0.6:
            recommendations.append({
                'category': 'Prevention',
                'action': 'Consider taking preventive medication if prescribed',
                'priority': 'high'
            })

        recommendations.append({
            'category': 'Hydration',
            'action': 'Ensure you stay well hydrated throughout the day',
            'priority': 'medium' if probability > 0.4 else 'low'
        })

        if features.get('stress_intensity', 0) > 5:
            recommendations.append({
                'category': 'Activity',
                'action': 'Light exercise or a short walk may help reduce stress',
                'priority': 'low'
            })

        return recommendations

    def save_model(self, filepath='migraine_weather_model.pkl'):
        """Save trained model to disk"""
        model_data = {
            'occurrence_model': self.occurrence_model,
            'scaler': self.scaler,
            'feature_importance': self.feature_importance,
            'feature_weights': self.feature_weights
        }
        joblib.dump(model_data, filepath)
        print(f"\n✅ Model saved to {filepath}")

    def load_model(self, filepath='migraine_weather_model.pkl'):
        """Load trained model from disk"""
        model_data = joblib.load(filepath)
        self.occurrence_model = model_data['occurrence_model']
        self.scaler = model_data['scaler']
        self.feature_importance = model_data['feature_importance']
        self.feature_weights = model_data['feature_weights']
        print(f"✅ Model loaded from {filepath}")
        return self


class ModelTrainer:
    """Handles the complete training pipeline with weather data"""

    @staticmethod
    def load_csv_data(filepath):
        """Load and preprocess CSV data"""
        print(f"📂 Loading data from {filepath}...")
        df = pd.read_csv(filepath, low_memory=False)
        print(f"✅ Loaded {len(df):,} records")
        print(f"📋 Columns found: {list(df.columns)[:10]}...")
        return df

    @staticmethod
    def validate_data(df):
        """Validate data quality and show statistics"""
        print("\n" + "=" * 60)
        print("DATA VALIDATION")
        print("=" * 60)
        print(f"Total records: {len(df):,}")

        required_cols = ['migraine', 'stress_intensity', 'sleep_duration']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"⚠️  WARNING: Missing required columns: {missing_cols}")

        # Check for weather columns
        weather_cols = ['temp_mean', 'pressure_mean', 'wind_mean', 'precip_mean']
        weather_found = [col for col in weather_cols if col in df.columns]
        print(f"\n🌦️  Weather columns found: {len(weather_found)}/{len(weather_cols)}")
        for col in weather_found:
            non_null = df[col].notna().sum()
            print(f"   • {col}: {non_null:,} records ({non_null / len(df) * 100:.1f}%)")

        if 'timestamp' in df.columns:
            try:
                dates = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')
                valid_dates = dates.dropna()
                if len(valid_dates) > 0:
                    print(f"\n📅 Date range: {valid_dates.min().date()} to {valid_dates.max().date()}")
                    print(f"   Duration: {(valid_dates.max() - valid_dates.min()).days} days")
            except Exception as e:
                print(f"⚠️  Could not parse dates: {e}")

        if 'migraine' in df.columns:
            migraine_numeric = pd.to_numeric(df['migraine'], errors='coerce')
            migraine_count = (migraine_numeric > 0).sum()
            total_valid = migraine_numeric.notna().sum()
            if total_valid > 0:
                print(f"\n🔴 Migraine Events: {migraine_count:,} ({migraine_count / total_valid * 100:.2f}%)")
                print(
                    f"🟢 Healthy Days: {total_valid - migraine_count:,} ({(total_valid - migraine_count) / total_valid * 100:.2f}%)")

        print("=" * 60)
        return True

    @staticmethod
    def train_model_from_csv(health_csv, weather_csv=None, gender='female',
                             save_path='migraine_weather_model.pkl', sample_size=None):
        """
        Complete training pipeline from CSV files

        Args:
            health_csv: Path to health data CSV
            weather_csv: Path to weather data CSV (optional)
            gender: 'female' or 'male'
            save_path: Where to save the trained model
            sample_size: Limit training data size (None for all data)
        """
        health_df = ModelTrainer.load_csv_data(health_csv)

        if weather_csv:
            weather_df = ModelTrainer.load_csv_data(weather_csv)
            model = MigrainePredictionModel()
            health_df = model.merge_weather_data(health_df, weather_df)
        else:
            print("⚠️  No weather data provided. Training without weather features.")

        ModelTrainer.validate_data(health_df)

        print("\n" + "=" * 60)
        print("MODEL TRAINING")
        print("=" * 60)
        model = MigrainePredictionModel()
        model.train(health_df, gender=gender, sample_size=sample_size)

        model.save_model(save_path)

        return model


# Example usage
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🧠 MIGRAINE PREDICTION ML MODEL WITH WEATHER DATA")
    print("=" * 60 + "\n")

    # Train with health data AND weather data
    model = ModelTrainer.train_model_from_csv(
        health_csv='synthetic_data_10_000/health_data_100000_90.csv',
        weather_csv='weather_data.csv',
        gender='female',
        save_path='migraine_weather_model.pkl',
        sample_size=100000
    )

    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE!")
    print("=" * 60)
    print("📦 Model saved with weather correlations analyzed")
    print("💡 Use model.predict(new_data) to make predictions")
    print("🌦️  Weather features integrated successfully")
    print("=" * 60)