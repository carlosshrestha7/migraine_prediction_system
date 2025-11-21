# predict_migraine.py
"""
Prediction script with enhanced recommendations
"""

import pandas as pd
from enhanced_migraine_model import EnhancedMigrainePredictionModel
from migraine_recommendation_engine import format_recommendations_for_display


def main():
    # Load the enhanced model
    print("Loading migraine prediction model...")
    model = EnhancedMigrainePredictionModel()
    model.load_model('migraine_model.pkl')
    
    # Example user data - replace with actual data
    sample_data = pd.DataFrame([{
        'time': '2024-01-15 14:00:00',
        'stressLevel.value': 7,
        'foodIntake.skippedBreakfast': 1,
        'mood.value': 3,
        'activity_index': 25,
        'intensity': 0,  # This would be the target in training, 0 for prediction
        'duration_m': 0   # This would be the target in training, 0 for prediction
    }])
    
    # User context - you can get this from user input
    user_symptoms = "light sensitivity, fatigue"
    user_triggers = "stress, missed meals"
    
    print("Making prediction with personalized recommendations...")
    
    # Make prediction with enhanced recommendations
    result = model.predict_with_recommendations(
        sample_data, 
        gender='female',
        user_symptoms=user_symptoms,
        user_triggers=user_triggers
    )
    
    # Display formatted results
    print("\n" + "=" * 50)
    print("MIGRAINE PREDICTION RESULTS")
    print("=" * 50)
    print(format_recommendations_for_display(result))
    print("=" * 50)


if __name__ == "__main__":
    main()