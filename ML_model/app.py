# app.py
"""
Simple CLI for Migraine Prediction System
"""

import pandas as pd
from enhanced_migraine_model import EnhancedMigrainePredictionModel, format_recommendations_for_display
import os

class MigrainePredictorApp:
    def __init__(self):
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            self.model = EnhancedMigrainePredictionModel()
            self.model.load_model('migraine_model.pkl')
            print("✅ Model loaded successfully!")
        except FileNotFoundError:
            print("❌ Model not found. Please run train_model.py first.")
            return False
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
        return True
    
    def get_user_input(self):
        """Get input from user for prediction"""
        print("\n" + "="*50)
        print("MIGRAINE PREDICTION INPUT")
        print("="*50)
        
        # Stress level
        stress = float(input("Enter stress level (0-10): ") or "5")
        
        # Mood
        mood = float(input("Enter mood level (0-10, where 10 is best): ") or "5")
        
        # Meal skipping
        skipped_breakfast = int(input("Skipped breakfast? (1 for yes, 0 for no): ") or "0")
        skipped_lunch = int(input("Skipped lunch? (1 for yes, 0 for no): ") or "0")
        skipped_dinner = int(input("Skipped dinner? (1 for yes, 0 for no): ") or "0")
        
        # Activity
        activity = float(input("Enter activity level (0-100): ") or "50")
        
        # Symptoms and triggers
        symptoms = input("Any current symptoms? (comma-separated, e.g., 'light sensitivity, fatigue'): ") or ""
        triggers = input("Known triggers today? (comma-separated, e.g., 'stress, missed meals'): ") or ""
        
        # Create input data
        user_data = pd.DataFrame([{
            'time': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stressLevel.value': stress,
            'mood.value': mood,
            'foodIntake.skippedBreakfast': skipped_breakfast,
            'foodIntake.skippedLunch': skipped_lunch,
            'foodIntake.skippedDinner': skipped_dinner,
            'activity_index': activity,
            'intensity': 0,  # Current migraine intensity (0 for prediction)
            'duration_m': 0
        }])
        
        return user_data, symptoms, triggers
    
    def predict(self):
        """Make prediction with user input"""
        if not self.model:
            print("❌ Model not loaded. Cannot make predictions.")
            return
        
        user_data, symptoms, triggers = self.get_user_input()
        
        print("\n🔄 Analyzing your data...")
        
        try:
            result = self.model.predict_with_recommendations(
                user_data,
                gender='female',
                user_symptoms=symptoms,
                user_triggers=triggers
            )
            
            print("\n" + "="*60)
            print("PREDICTION RESULTS")
            print("="*60)
            print(format_recommendations_for_display(result))
            print("="*60)
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
    
    def batch_predict(self, csv_file):
        """Make predictions from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} records from {csv_file}")
            
            results = []
            for i, row in df.iterrows():
                user_data = pd.DataFrame([row])
                result = self.model.predict_with_recommendations(user_data, gender='female')
                results.append(result)
                
                if i < 3:  # Show first 3 results
                    print(f"\n--- Prediction {i+1} ---")
                    print(f"Risk: {result['risk_label']} ({result['probability']}%)")
            
            return results
            
        except Exception as e:
            print(f"❌ Batch prediction error: {e}")
    
    def run(self):
        """Main application loop"""
        if not self.load_model():
            return
        
        while True:
            print("\n" + "="*40)
            print("MIGRAINE PREDICTION SYSTEM")
            print("="*40)
            print("1. Make Prediction")
            print("2. Batch Predict from CSV")
            print("3. Retrain Model")
            print("4. Exit")
            
            choice = input("\nChoose option (1-4): ").strip()
            
            if choice == '1':
                self.predict()
            elif choice == '2':
                csv_file = input("Enter CSV file path: ").strip()
                if os.path.exists(csv_file):
                    self.batch_predict(csv_file)
                else:
                    print("❌ File not found!")
            elif choice == '3':
                print("🔄 Retraining model...")
                os.system('python train_model.py')
            elif choice == '4':
                print("👋 Goodbye!")
                break
            else:
                print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    app = MigrainePredictorApp()
    app.run()