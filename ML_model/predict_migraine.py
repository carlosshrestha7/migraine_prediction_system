# predict_migraine.py
"""
Prediction script with enhanced recommendations - UPDATED for your data
"""

import pandas as pd
from enhanced_migraine_model import EnhancedMigrainePredictionModel, format_recommendations_for_display


def main():
    # Load the enhanced model
    print("Loading migraine prediction model...")
    model = EnhancedMigrainePredictionModel()
    
    try:
        model.load_model('migraine_model.pkl')
        print("✅ Model loaded successfully!")
    except FileNotFoundError:
        print("❌ Model file not found. Please train the model first using train_model.py")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Example user data - using YOUR actual column names
    sample_data = pd.DataFrame([{
        'time': '2024-01-15 14:00:00',
        'stressLevel.value': 7,
        'mood.value': 3,
        'foodIntake.skippedBreakfast': 1,
        'foodIntake.skippedLunch': 0,
        'foodIntake.skippedDinner': 0,
        'activity_index': 25,
        'intensity': 0,  # No current migraine
        'duration_m': 0,
        'predictedStress': 6.5
    }])
    
    # User context
    user_symptoms = "light sensitivity, fatigue"
    user_triggers = "stress, missed meals"
    
    print("Making prediction with personalized recommendations...")
    
    try:
        # Make prediction with enhanced recommendations
        result = model.predict_with_recommendations(
            sample_data, 
            gender='female',
            user_symptoms=user_symptoms,
            user_triggers=user_triggers
        )
        
        # Display formatted results
        print("\n" + "=" * 60)
        print("MIGRAINE PREDICTION RESULTS")
        print("=" * 60)
        print(format_recommendations_for_display(result))
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ Error during prediction: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()