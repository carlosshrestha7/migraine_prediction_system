# predict_migraine.py
"""
Prediction script with enhanced recommendations
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
    
    # Example user data - using column names that match your training data
    sample_data = pd.DataFrame([{
        'timestamp': 1705320000,  # Example timestamp
        'stress_intensity': 7,
        'sleep_duration': 6.5,
        'sleep_deficit': 1,
        'missed_meal': 1,
        'menstruation': 0,
        'delivery': 0,
        'migraine_days_per_month': 8,
        'p_stress': 0.7,
        'p_hormones': 0.2,
        'p_sleep': 0.6,
        'p_weather': 0.3,
        'p_meals': 0.8,
        'migraine_probability': 0.65,
        'migraine': 0  # This would be the target in training, 0 for prediction
    }])
    
    # User context - you can get this from user input
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
        print("💡 Tip: Make sure your input data matches the training data format")


if __name__ == "__main__":
    main()
    