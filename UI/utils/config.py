# UI/utils/config.py - APP CONFIGURATION
APP_CONFIG = {
    'APP_TITLE': 'Migraine Prediction System',
    'APP_ICON': '🧠',
    'APP_DESCRIPTION': 'Track your daily factors and predict migraine risk',
    'LAYOUT': 'wide',
    'SIDEBAR_STATE': 'expanded',
    
    # Risk bands from Workflow Flowchart
    'RISK_BANDS': {
        'green': {'min': 0, 'max': 20, 'label': 'Low Risk', 'color': '🟢'},
        'yellow': {'min': 20, 'max': 50, 'label': 'Moderate Risk', 'color': '🟡'},
        'orange': {'min': 50, 'max': 70, 'label': 'High Risk', 'color': '🟠'},
        'red': {'min': 70, 'max': 100, 'label': 'Very High Risk', 'color': '🔴'}
    },
    
    # From Comprehensive Trigger Categories
    'SYMPTOMS': [
        "Headache", "Aura", "Nausea", "Light Sensitivity", 
        "Sound Sensitivity", "Visual Disturbances", "Dizziness"
    ],
    
    'TRIGGERS': [
        "Temperature Changes", "Humidity Levels", "Sleep Patterns",
        "Physical Activity", "Meal Timing", "Hydration Levels",
        "Stress Patterns", "Strong Smells", "Bright Lights",
        "Loud Noises", "Hormonal Changes", "Travel/Jet Lag"
    ]
}