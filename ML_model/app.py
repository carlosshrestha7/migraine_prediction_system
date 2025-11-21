# app.py
"""
Enhanced CLI for Migraine Prediction System with Voice, Colors, and Visualizations
"""

import pandas as pd
from enhanced_migraine_model import EnhancedMigrainePredictionModel, format_recommendations_for_display
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import random

try:
    import speech_recognition as sr
    VOICE_ENABLED = True
except ImportError:
    VOICE_ENABLED = False

class Color:
    """ANSI color codes for terminal output"""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    ORANGE = '\033[33m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class MigrainePredictorApp:
    def __init__(self):
        self.model = None
        self.recognizer = sr.Recognizer() if VOICE_ENABLED else None
        self.history = []
        self.load_model()
    
    def load_model(self):
        """Load the trained model with colorful output"""
        try:
            print(f"{Color.CYAN}🌀 Loading migraine prediction model...{Color.END}")
            self.model = EnhancedMigrainePredictionModel()
            self.model.load_model('migraine_model.pkl')
            print(f"{Color.GREEN}✅ Model loaded successfully!{Color.END}")
        except FileNotFoundError:
            print(f"{Color.RED}❌ Model not found. Please run train_model.py first.{Color.END}")
            return False
        except Exception as e:
            print(f"{Color.RED}❌ Error loading model: {e}{Color.END}")
            return False
        return True
    
    def voice_input(self, prompt):
        """Get voice input from user"""
        if not VOICE_ENABLED:
            print(f"{Color.YELLOW}⚠️  Voice recognition not available. Install speechrecognition and pyaudio.{Color.END}")
            return None
            
        try:
            with sr.Microphone() as source:
                print(f"{Color.CYAN}🎤 {prompt} (speak now)...{Color.END}")
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                audio = self.recognizer.listen(source, timeout=10)
                
            text = self.recognizer.recognize_google(audio)
            print(f"{Color.GREEN}🗣️  You said: {text}{Color.END}")
            return text.lower()
        except sr.UnknownValueError:
            print(f"{Color.RED}❌ Could not understand audio{Color.END}")
        except sr.RequestError as e:
            print(f"{Color.RED}❌ Speech recognition error: {e}{Color.END}")
        except sr.WaitTimeoutError:
            print(f"{Color.YELLOW}⏰ No speech detected{Color.END}")
        
        return None
    
    def get_risk_color(self, risk_band):
        """Get color code for risk level"""
        colors = {
            'green': Color.GREEN,
            'yellow': Color.YELLOW,
            'orange': Color.ORANGE,
            'red': Color.RED
        }
        return colors.get(risk_band, Color.WHITE)
    
    def display_risk_meter(self, probability):
        """Display a visual risk meter"""
        risk_levels = [
            (20, "🟢 LOW", Color.GREEN),
            (50, "🟡 MODERATE", Color.YELLOW),
            (70, "🟠 HIGH", Color.ORANGE),
            (100, "🔴 VERY HIGH", Color.RED)
        ]
        
        print(f"\n{Color.BOLD}📊 MIGRAINE RISK METER:{Color.END}")
        print("┌" + "─" * 50 + "┐")
        
        for max_prob, label, color in risk_levels:
            marker = "█" if probability <= max_prob else "░"
            if probability <= max_prob:
                current_indicator = f"{color}← Your Risk{Color.END}" if probability <= max_prob and probability > (max_prob - 20) else ""
                break
        
        # Create visual meter
        meter_length = 50
        filled = int((probability / 100) * meter_length)
        meter_bar = f"{Color.RED}{'█' * filled}{Color.END}{Color.WHITE}{'░' * (meter_length - filled)}{Color.END}"
        
        print(f"│ {meter_bar} │")
        print(f"│ {Color.WHITE}0%{' ' * 22}{probability}%{' ' * 22}100%{Color.END} │")
        print("└" + "─" * 50 + "┘")
    
    def get_user_input(self):
        """Get input from user for prediction with voice option"""
        print(f"\n{Color.CYAN}{'='*50}{Color.END}")
        print(f"{Color.BOLD}🎯 MIGRAINE PREDICTION INPUT{Color.END}")
        print(f"{Color.CYAN}{'='*50}{Color.END}")
        
        print(f"\n{Color.BLUE}💡 You can speak your answers or type them{Color.END}")
        
        # Voice or text mode
        use_voice = input(f"\n{Color.CYAN}Use voice input? (y/n): {Color.END}").lower().strip() == 'y'
        
        inputs = {}
        
        # Stress level
        if use_voice:
            voice_response = self.voice_input("On a scale of 0 to 10, how stressed are you?")
            if voice_response:
                try:
                    # Simple voice parsing
                    if 'zero' in voice_response or '0' in voice_response:
                        stress = 0
                    elif 'one' in voice_response or '1' in voice_response:
                        stress = 1
                    # ... add more parsing as needed
                    else:
                        # Extract number from voice
                        stress = float(''.join(filter(str.isdigit, voice_response)) or 5)
                except:
                    stress = 5
            else:
                stress = float(input(f"{Color.CYAN}Enter stress level (0-10): {Color.END}") or "5")
        else:
            stress = float(input(f"{Color.CYAN}Enter stress level (0-10): {Color.END}") or "5")
        
        inputs['stress'] = max(0, min(10, stress))
        
        # Mood
        if use_voice:
            voice_response = self.voice_input("On a scale of 0 to 10, how is your mood? 10 being best")
            if voice_response:
                try:
                    mood = float(''.join(filter(str.isdigit, voice_response)) or 5)
                except:
                    mood = 5
            else:
                mood = float(input(f"{Color.CYAN}Enter mood level (0-10, where 10 is best): {Color.END}") or "5")
        else:
            mood = float(input(f"{Color.CYAN}Enter mood level (0-10, where 10 is best): {Color.END}") or "5")
        
        inputs['mood'] = max(0, min(10, mood))
        
        # Meal skipping
        questions = [
            ("Did you skip breakfast?", "skipped_breakfast"),
            ("Did you skip lunch?", "skipped_lunch"), 
            ("Did you skip dinner?", "skipped_dinner")
        ]
        
        for question, key in questions:
            if use_voice:
                voice_response = self.voice_input(question)
                if voice_response:
                    inputs[key] = 1 if any(word in voice_response for word in ['yes', 'yeah', 'yep', '1']) else 0
                else:
                    inputs[key] = int(input(f"{Color.CYAN}{question} (1 for yes, 0 for no): {Color.END}") or "0")
            else:
                inputs[key] = int(input(f"{Color.CYAN}{question} (1 for yes, 0 for no): {Color.END}") or "0")
        
        # Activity
        if use_voice:
            voice_response = self.voice_input("What's your activity level from 0 to 100?")
            if voice_response:
                try:
                    activity = float(''.join(filter(str.isdigit, voice_response)) or 50)
                except:
                    activity = 50
            else:
                activity = float(input(f"{Color.CYAN}Enter activity level (0-100): {Color.END}") or "50")
        else:
            activity = float(input(f"{Color.CYAN}Enter activity level (0-100): {Color.END}") or "50")
        
        inputs['activity'] = max(0, min(100, activity))
        
        # Symptoms and triggers
        symptoms = input(f"{Color.CYAN}Any current symptoms? (comma-separated): {Color.END}") or ""
        triggers = input(f"{Color.CYAN}Known triggers today? (comma-separated): {Color.END}") or ""
        
        # Create input data
        user_data = pd.DataFrame([{
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stressLevel.value': inputs['stress'],
            'mood.value': inputs['mood'],
            'foodIntake.skippedBreakfast': inputs['skipped_breakfast'],
            'foodIntake.skippedLunch': inputs['skipped_lunch'],
            'foodIntake.skippedDinner': inputs['skipped_dinner'],
            'activity_index': inputs['activity'],
            'intensity': 0,
            'duration_m': 0
        }])
        
        return user_data, symptoms, triggers, inputs
    
    def create_visualization(self, result, user_inputs):
        """Create a visualization of the prediction results"""
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Migraine Risk Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Risk Probability Gauge
            risk_levels = ['Low', 'Moderate', 'High', 'Very High']
            risk_thresholds = [20, 50, 70, 100]
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            
            current_risk = result['probability']
            risk_index = next(i for i, threshold in enumerate(risk_thresholds) if current_risk <= threshold)
            
            ax1.barh(risk_levels, risk_thresholds, color=colors, alpha=0.3)
            ax1.barh(risk_levels[risk_index], risk_thresholds[risk_index], 
                     color=colors[risk_index], alpha=0.7)
            ax1.axvline(current_risk, color='black', linestyle='--', alpha=0.8)
            ax1.text(current_risk + 2, risk_index, f'{current_risk}%', va='center', fontweight='bold')
            ax1.set_xlabel('Probability (%)')
            ax1.set_title('Migraine Risk Level')
            
            # 2. Factor Analysis
            factors = {
                'Stress': user_inputs.get('stress', 0),
                'Mood': 10 - user_inputs.get('mood', 5),  # Invert mood (lower mood = higher risk)
                'Meals Skipped': sum([user_inputs.get('skipped_breakfast', 0), 
                                    user_inputs.get('skipped_lunch', 0), 
                                    user_inputs.get('skipped_dinner', 0)]) * 3.33,
                'Low Activity': max(0, 50 - user_inputs.get('activity', 50)) / 5
            }
            
            factor_names = list(factors.keys())
            factor_values = list(factors.values())
            colors = ['#e74c3c', '#9b59b6', '#3498db', '#f1c40f']
            
            bars = ax2.bar(factor_names, factor_values, color=colors, alpha=0.7)
            ax2.set_ylabel('Risk Contribution')
            ax2.set_title('Risk Factor Analysis')
            ax2.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, factor_values):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{value:.1f}', ha='center', va='bottom')
            
            # 3. Risk Trend (simulated)
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            trend_data = [random.randint(10, 90) for _ in range(6)] + [current_risk]
            
            ax3.plot(days, trend_data, marker='o', linewidth=2, markersize=8, color='#3498db')
            ax3.fill_between(days, trend_data, alpha=0.3, color='#3498db')
            ax3.set_ylim(0, 100)
            ax3.set_ylabel('Risk Probability (%)')
            ax3.set_title('Weekly Risk Trend')
            ax3.grid(True, alpha=0.3)
            
            # Highlight current day
            ax3.plot(days[-1], current_risk, marker='o', markersize=12, color='#e74c3c')
            
            # 4. Recommendations Priority
            recommendations = result.get('key_recommendations', [])
            if recommendations:
                rec_names = [rec['action'][:30] + '...' if len(rec['action']) > 30 else rec['action'] 
                           for rec in recommendations]
                urgency_map = {'high': 3, 'medium': 2, 'low': 1}
                urgency_values = [urgency_map.get(rec.get('urgency', 'low'), 1) for rec in recommendations]
                urgency_colors = ['#e74c3c', '#f39c12', '#2ecc71']
                
                bars = ax4.barh(rec_names, urgency_values, color=[urgency_colors[u-1] for u in urgency_values])
                ax4.set_xlabel('Priority (1=Low, 3=High)')
                ax4.set_title('Recommended Actions')
            
            plt.tight_layout()
            plt.savefig('migraine_analysis.png', dpi=150, bbox_inches='tight')
            print(f"{Color.GREEN}📊 Visualization saved as 'migraine_analysis.png'{Color.END}")
            plt.show()
            
        except Exception as e:
            print(f"{Color.YELLOW}⚠️  Could not create visualization: {e}{Color.END}")
    
    def predict(self):
        """Make prediction with user input and enhanced display"""
        if not self.model:
            print(f"{Color.RED}❌ Model not loaded. Cannot make predictions.{Color.END}")
            return
        
        user_data, symptoms, triggers, user_inputs = self.get_user_input()
        
        print(f"\n{Color.CYAN}🔄 Analyzing your data with AI...{Color.END}")
        
        try:
            result = self.model.predict_with_recommendations(
                user_data,
                gender='female',
                user_symptoms=symptoms,
                user_triggers=triggers
            )
            
            # Store in history
            self.history.append({
                'timestamp': datetime.now(),
                'result': result,
                'inputs': user_inputs
            })
            
            # Display results with colors
            risk_color = self.get_risk_color(result['risk_band'])
            
            print(f"\n{Color.CYAN}{'='*60}{Color.END}")
            print(f"{Color.BOLD}🎯 MIGRAINE PREDICTION RESULTS{Color.END}")
            print(f"{Color.CYAN}{'='*60}{Color.END}")
            
            # Risk meter
            self.display_risk_meter(result['probability'])
            
            # Main result
            print(f"\n{risk_color}{Color.BOLD}📈 RISK LEVEL: {result['risk_label']} ({result['probability']}% probability){Color.END}")
            print(f"{Color.WHITE}💡 {result['risk_message']}{Color.END}")
            
            # Enhanced display of recommendations
            print(f"\n{Color.BOLD}🛡️  PERSONALIZED RECOMMENDATIONS:{Color.END}")
            for i, rec in enumerate(result.get('key_recommendations', []), 1):
                urgency_color = {
                    'high': Color.RED,
                    'medium': Color.YELLOW, 
                    'low': Color.GREEN
                }.get(rec.get('urgency', 'low'), Color.WHITE)
                
                print(f"\n{urgency_color}{i}. {rec['action']}{Color.END}")
                print(f"   {Color.CYAN}💡 Why: {rec['reason']}{Color.END}")
            
            # Quick actions
            print(f"\n{Color.BOLD}⚡ QUICK ACTIONS:{Color.END}")
            for action in result.get('quick_actions', []):
                print(f"   {Color.GREEN}• {action}{Color.END}")
            
            # Prevention tips
            print(f"\n{Color.BOLD}🎯 PREVENTION TIPS:{Color.END}")
            for tip in result.get('prevention_tips', []):
                print(f"   {Color.BLUE}• {tip}{Color.END}")
            
            # Risk factors
            if result.get('top_risk_factors'):
                print(f"\n{Color.BOLD}⚠️  IDENTIFIED RISK FACTORS:{Color.END}")
                for factor in result['top_risk_factors']:
                    print(f"   {Color.ORANGE}• {factor}{Color.END}")
            
            print(f"{Color.CYAN}{'='*60}{Color.END}")
            
            # Ask to show visualization
            show_viz = input(f"\n{Color.CYAN}Show detailed visualization? (y/n): {Color.END}").lower().strip()
            if show_viz == 'y':
                self.create_visualization(result, user_inputs)
                
        except Exception as e:
            print(f"{Color.RED}❌ Prediction error: {e}{Color.END}")
            import traceback
            traceback.print_exc()
    
    def show_history(self):
        """Show prediction history"""
        if not self.history:
            print(f"{Color.YELLOW}No prediction history yet.{Color.END}")
            return
            
        print(f"\n{Color.BOLD}📈 PREDICTION HISTORY ({len(self.history)} entries):{Color.END}")
        for i, entry in enumerate(self.history[-5:], 1):  # Show last 5
            result = entry['result']
            risk_color = self.get_risk_color(result['risk_band'])
            print(f"{i}. {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                  f"{risk_color}{result['risk_label']} ({result['probability']}%){Color.END}")
    
    def demo_mode(self):
        """Run a demo with sample data"""
        print(f"\n{Color.MAGENTA}{'='*50}{Color.END}")
        print(f"{Color.BOLD}🎬 DEMO MODE - Sample Prediction{Color.END}")
        print(f"{Color.MAGENTA}{'='*50}{Color.END}")
        
        sample_data = pd.DataFrame([{
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stressLevel.value': 8,
            'mood.value': 3,
            'foodIntake.skippedBreakfast': 1,
            'foodIntake.skippedLunch': 0,
            'foodIntake.skippedDinner': 0,
            'activity_index': 25,
            'intensity': 0,
            'duration_m': 0
        }])
        
        user_inputs = {
            'stress': 8, 'mood': 3, 'skipped_breakfast': 1,
            'skipped_lunch': 0, 'skipped_dinner': 0, 'activity': 25
        }
        
        print(f"{Color.CYAN}📊 Demo Input: High stress (8/10), Low mood (3/10), Skipped breakfast, Low activity{Color.END}")
        
        try:
            result = self.model.predict_with_recommendations(
                sample_data,
                gender='female',
                user_symptoms="light sensitivity, fatigue",
                user_triggers="stress, missed meals"
            )
            
            risk_color = self.get_risk_color(result['risk_band'])
            print(f"\n{risk_color}{Color.BOLD}🎯 DEMO RESULT: {result['risk_label']} ({result['probability']}% probability){Color.END}")
            
            # Show visualization for demo
            self.create_visualization(result, user_inputs)
            
        except Exception as e:
            print(f"{Color.RED}❌ Demo error: {e}{Color.END}")
    
    def run(self):
        """Main application loop"""
        if not self.load_model():
            return
        
        while True:
            print(f"\n{Color.CYAN}{'='*40}{Color.END}")
            print(f"{Color.BOLD}🧠 MIGRAINE PREDICTION SYSTEM{Color.END}")
            print(f"{Color.CYAN}{'='*40}{Color.END}")
            print(f"{Color.GREEN}1. Make Prediction{Color.END}")
            print(f"{Color.BLUE}2. Batch Predict from CSV{Color.END}")
            print(f"{Color.YELLOW}3. Show History{Color.END}")
            print(f"{Color.MAGENTA}4. Demo Mode{Color.END}")
            print(f"{Color.ORANGE}5. Retrain Model{Color.END}")
            print(f"{Color.RED}6. Exit{Color.END}")
            
            choice = input(f"\n{Color.WHITE}Choose option (1-6): {Color.END}").strip()
            
            if choice == '1':
                self.predict()
            elif choice == '2':
                csv_file = input(f"{Color.CYAN}Enter CSV file path: {Color.END}").strip()
                if os.path.exists(csv_file):
                    self.batch_predict(csv_file)
                else:
                    print(f"{Color.RED}❌ File not found!{Color.END}")
            elif choice == '3':
                self.show_history()
            elif choice == '4':
                self.demo_mode()
            elif choice == '5':
                print(f"{Color.YELLOW}🔄 Retraining model...{Color.END}")
                os.system('python train_model.py')
            elif choice == '6':
                print(f"{Color.GREEN}👋 Thank you for using Migraine Prediction System!{Color.END}")
                break
            else:
                print(f"{Color.RED}❌ Invalid choice. Please try again.{Color.END}")

if __name__ == "__main__":
    app = MigrainePredictorApp()
    app.run()