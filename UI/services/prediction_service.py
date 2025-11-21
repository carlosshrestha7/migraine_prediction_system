# UI/services/prediction_service.py - MOCK PREDICTION SERVICE (Matches ML Service)
from UI.utils.config import APP_CONFIG
import random
from datetime import datetime

def get_prediction(user_data):
    """Get migraine prediction - matches your ML model predict() function"""
    
    # Calculate mock probability based on user inputs
    probability = calculate_mock_probability(user_data)
    
    # Determine risk band
    risk_band = determine_risk_band(probability)
    risk_info = APP_CONFIG['RISK_BANDS'][risk_band]
    
    # Generate recommendations
    recommendations = generate_recommendations(user_data, probability)
    
    # Return prediction matching your ML model structure
    return {
        'probability': probability,
        'risk_band': risk_band,
        'risk_label': risk_info['label'],
        'risk_message': generate_risk_message(user_data, probability),
        'predicted_intensity': random.uniform(0, 8),
        'intensity_label': predict_intensity(probability),
        'predicted_duration_minutes': random.randint(30, 480),
        'recommendations': recommendations,
        'timestamp': datetime.now().isoformat()
    }

def calculate_mock_probability(user_data):
    """Calculate mock probability based on user inputs"""
    base_risk = 10  # Base risk
    
    # Stress contribution (20% weight from your model)
    stress_contrib = user_data.get('stress_level', 5) * 2
    
    # Sleep contribution (25% weight)
    sleep_contrib = max(0, 8 - user_data.get('sleep_hours', 7)) * 3
    
    # Mood contribution (10% weight)  
    mood_contrib = max(0, 10 - user_data.get('mood', 7)) * 1.5
    
    # Meal contribution (15% weight)
    meals_skipped = sum([
        1 for meal in ['had_breakfast', 'had_lunch', 'had_dinner'] 
        if not user_data.get(meal, True)
    ])
    meal_contrib = meals_skipped * 5
    
    # Symptoms contribution
    symptom_contrib = len(user_data.get('symptoms', [])) * 3
    
    # Triggers contribution
    trigger_contrib = len(user_data.get('triggers', [])) * 2
    
    total_risk = (base_risk + stress_contrib + sleep_contrib + 
                 mood_contrib + meal_contrib + symptom_contrib + trigger_contrib)
    
    return min(round(total_risk, 1), 95)

def determine_risk_band(probability):
    """Determine risk band based on probability"""
    for band, info in APP_CONFIG['RISK_BANDS'].items():
        if info['min'] <= probability < info['max']:
            return band
    return 'red'

def predict_intensity(probability):
    """Predict migraine intensity"""
    if probability < 30:
        return "None"
    elif probability < 50:
        return "Light" 
    elif probability < 70:
        return "Medium"
    else:
        return "Strong"

def generate_risk_message(user_data, probability):
    """Generate personalized risk message"""
    factors = []
    
    if user_data.get('stress_level', 0) > 6:
        factors.append("high stress")
    if user_data.get('sleep_hours', 7) < 6:
        factors.append("insufficient sleep")
    if user_data.get('mood', 7) < 5:
        factors.append("low mood")
    
    if factors:
        return f"Elevated risk due to {', '.join(factors)}"
    else:
        return "Low risk based on current factors"

def generate_recommendations(user_data, probability):
    """Generate personalized recommendations"""
    recommendations = []
    
    # Stress management
    if user_data.get('stress_level', 0) > 6:
        recommendations.append({
            'category': 'Stress Management',
            'action': 'Practice deep breathing or meditation',
            'priority': 'high'
        })
    
    # Sleep recommendations
    if user_data.get('sleep_hours', 7) < 6:
        recommendations.append({
            'category': 'Sleep',
            'action': 'Consider a short nap or early bedtime',
            'priority': 'high'
        })
    
    # Nutrition recommendations
    meals_skipped = sum([
        1 for meal in ['had_breakfast', 'had_lunch', 'had_dinner'] 
        if not user_data.get(meal, True)
    ])
    if meals_skipped > 0:
        recommendations.append({
            'category': 'Nutrition', 
            'action': 'Eat regular meals and stay hydrated',
            'priority': 'medium'
        })
    
    # Activity recommendations
    if user_data.get('activity_level', 50) < 30:
        recommendations.append({
            'category': 'Activity',
            'action': 'Light exercise may help improve mood',
            'priority': 'low'
        })
    
    # General hydration
    recommendations.append({
        'category': 'Hydration',
        'action': 'Drink plenty of water throughout the day',
        'priority': 'medium'
    })
    
    return recommendations