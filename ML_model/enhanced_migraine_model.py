# enhanced_migraine_model.py
"""
Enhanced migraine prediction model with recommendation engine
Combines ML predictions with actionable recommendations
"""

import pandas as pd
from migraine_model import MigrainePredictionModel
from migraine_recommendation_engine import MigraineRecommendationEngine, format_recommendations_for_display


class EnhancedMigrainePredictionModel(MigrainePredictionModel):
    """
    Enhanced model with improved recommendation engine
    """
    
    def __init__(self):
        super().__init__()
        self.recommendation_engine = MigraineRecommendationEngine()
    
    def predict_with_recommendations(self, user_data, gender='female', user_symptoms=None, user_triggers=None):
        """
        Enhanced prediction with detailed, personalized recommendations
        
        Args:
            user_data: DataFrame with user features
            gender: User gender for personalized recommendations
            user_symptoms: Comma-separated string of user symptoms
            user_triggers: Comma-separated string of known triggers
        
        Returns:
            Dictionary with predictions and personalized recommendations
        """
        try:
            # Get base prediction
            base_prediction = self.predict(user_data, gender)
            
            # Engineer features for context analysis
            features = self.engineer_features(user_data, gender)
            
            # Generate personalized recommendations
            if len(features) > 0:
                recommendations = self.recommendation_engine.generate_personalized_recommendations(
                    base_prediction, 
                    features.iloc[0],
                    user_symptoms,
                    user_triggers
                )
            else:
                # Fallback if no features engineered
                recommendations = {
                    'risk_level': base_prediction['risk_band'],
                    'focus_area': 'General Prevention',
                    'key_recommendations': [],
                    'quick_actions': [
                        "Ensure proper hydration",
                        "Maintain regular meal schedule",
                        "Manage stress levels"
                    ],
                    'prevention_tips': [
                        "💧 Stay hydrated throughout the day",
                        "🍽️ Eat balanced meals regularly",
                        "😴 Prioritize quality sleep"
                    ]
                }
            
            # Merge recommendations with base prediction
            enhanced_output = {**base_prediction, **recommendations}
            
            # Add predicted intensity and duration for compatibility
            if 'predicted_intensity' not in enhanced_output:
                enhanced_output['predicted_intensity'] = self._calculate_predicted_intensity(base_prediction['probability'])
                enhanced_output['intensity_label'] = self._get_intensity_label(enhanced_output['predicted_intensity'])
                enhanced_output['predicted_duration_minutes'] = self._estimate_duration(enhanced_output['predicted_intensity'])
            
            return enhanced_output
            
        except Exception as e:
            print(f"Error in enhanced prediction: {e}")
            # Return basic prediction if enhanced fails
            return self.predict(user_data, gender)
    
    def _calculate_predicted_intensity(self, probability):
        """Calculate predicted intensity based on probability"""
        # Scale probability (0-100) to intensity (0-10)
        if probability < 20:
            return 2
        elif probability < 50:
            return 4
        elif probability < 70:
            return 6
        elif probability < 85:
            return 8
        else:
            return 9
    
    def _get_intensity_label(self, intensity):
        """Get intensity label based on numeric value"""
        if intensity <= 3:
            return "Mild"
        elif intensity <= 6:
            return "Moderate"
        elif intensity <= 8:
            return "Severe"
        else:
            return "Very Severe"
    
    def _estimate_duration(self, intensity):
        """Estimate duration based on intensity"""
        # Base duration in minutes, increases with intensity
        base_duration = 120  # 2 hours base
        return base_duration + (intensity * 30)  # +30 minutes per intensity point


# Export the display function for easy access
format_recommendations_for_display = format_recommendations_for_display


# Simple test function
def test_enhanced_model():
    """Test the enhanced model with sample data"""
    try:
        model = EnhancedMigrainePredictionModel()
        model.load_model('migraine_model.pkl')
        
        # Sample test data
        sample_data = pd.DataFrame([{
            'time': '2024-01-15 14:00:00',
            'stressLevel.value': 7,
            'mood.value': 3,
            'foodIntake.skippedBreakfast': 1,
            'foodIntake.skippedLunch': 0,
            'foodIntake.skippedDinner': 0,
            'activity_index': 25,
            'intensity': 0,
            'duration_m': 0,
            'predictedStress': 6.5
        }])
        
        result = model.predict_with_recommendations(
            sample_data,
            gender='female',
            user_symptoms="light sensitivity, fatigue",
            user_triggers="stress, missed meals"
        )
        
        print("Enhanced model test successful!")
        print(f"Risk Level: {result['risk_label']}")
        print(f"Probability: {result['probability']}%")
        print(f"Key Recommendations: {len(result.get('key_recommendations', []))}")
        
        return True
        
    except Exception as e:
        print(f"Enhanced model test failed: {e}")
        return False


if __name__ == "__main__":
    test_enhanced_model()