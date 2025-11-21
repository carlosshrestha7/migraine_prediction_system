# enhanced_migraine_model.py
"""
Enhanced migraine prediction model with recommendation engine
Combines ML predictions with actionable recommendations
"""

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
        
        return enhanced_output


# Export the display function for easy access
format_recommendations_for_display = format_recommendations_for_display