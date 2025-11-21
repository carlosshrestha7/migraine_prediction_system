# predict_migraine.py
"""
Enhanced prediction script with colors and better formatting
"""

import pandas as pd
import sys
import os

# Add current directory to path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_migraine_model import EnhancedMigrainePredictionModel, format_recommendations_for_display

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ORANGE = '\033[33m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'

def colorize_risk(risk_band, text):
    colors = {
        'green': Color.GREEN,
        'yellow': Color.YELLOW, 
        'orange': Color.ORANGE,
        'red': Color.RED
    }
    return f"{colors.get(risk_band, Color.CYAN)}{text}{Color.END}"

def main():
    # Load the enhanced model
    print(f"{Color.CYAN}🌀 Loading migraine prediction model...{Color.END}")
    model = EnhancedMigrainePredictionModel()
    
    try:
        model.load_model('migraine_model.pkl')
        print(f"{Color.GREEN}✅ Model loaded successfully!{Color.END}")
    except FileNotFoundError:
        print(f"{Color.RED}❌ Model file not found. Please train the model first using train_model.py{Color.END}")
        return
    except Exception as e:
        print(f"{Color.RED}❌ Error loading model: {e}{Color.END}")
        return
    
    # Example user data
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
    
    # User context
    user_symptoms = "light sensitivity, fatigue"
    user_triggers = "stress, missed meals"
    
    print(f"{Color.BLUE}🎯 Making prediction with personalized recommendations...{Color.END}")
    
    try:
        # Make prediction with enhanced recommendations
        result = model.predict_with_recommendations(
            sample_data, 
            gender='female',
            user_symptoms=user_symptoms,
            user_triggers=user_triggers
        )
        
        # Display formatted results with colors
        risk_color = result['risk_band']
        
        print(f"\n{Color.CYAN}{'='*60}{Color.END}")
        print(f"{Color.BOLD}🧠 MIGRAINE PREDICTION RESULTS{Color.END}")
        print(f"{Color.CYAN}{'='*60}{Color.END}")
        
        # Risk header with color
        risk_display = colorize_risk(risk_color, f"📊 RISK LEVEL: {result['risk_label']} ({result['probability']}% probability)")
        print(f"\n{risk_display}")
        print(f"{Color.CYAN}💡 {result['risk_message']}{Color.END}")
        
        # Enhanced output
        formatted_output = format_recommendations_for_display(result)
        
        # Colorize the output
        for line in formatted_output.split('\n'):
            if '🟢' in line:
                print(Color.GREEN + line + Color.END)
            elif '🟡' in line:
                print(Color.YELLOW + line + Color.END)
            elif '🟠' in line:
                print(Color.ORANGE + line + Color.END)
            elif '🔴' in line:
                print(Color.RED + line + Color.END)
            elif '🎯' in line or '⚡' in line or '🛡️' in line or '⚠️' in line:
                print(Color.BLUE + line + Color.END)
            else:
                print(line)
                
        print(f"{Color.CYAN}{'='*60}{Color.END}")
        
    except Exception as e:
        print(f"{Color.RED}❌ Error during prediction: {e}{Color.END}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()