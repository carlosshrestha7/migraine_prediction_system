# app.py
"""
Enhanced CLI for Migraine Prediction System with Voice Input & Output
"""

import pandas as pd
import os
import sys
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import random
import threading
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from enhanced_migraine_model import EnhancedMigrainePredictionModel, format_recommendations_for_display

# Global voice flags - define them at module level
VOICE_INPUT_ENABLED = False
VOICE_OUTPUT_ENABLED = False

try:
    import speech_recognition as sr
    VOICE_INPUT_ENABLED = True
except ImportError:
    VOICE_INPUT_ENABLED = False
    print("⚠️  SpeechRecognition not available. Voice input disabled.")

try:
    import pyttsx3
    VOICE_OUTPUT_ENABLED = True
except ImportError:
    VOICE_OUTPUT_ENABLED = False
    print("⚠️  pyttsx3 not available. Voice output disabled.")

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

class VoiceAssistant:
    """Handles text-to-speech and speech-to-text functionality"""
    
    def __init__(self):
        self.tts_engine = None
        self.recognizer = None
        self.is_speaking = False
        
        # Use the global voice flags
        global VOICE_INPUT_ENABLED, VOICE_OUTPUT_ENABLED
        
        # Initialize TTS
        if VOICE_OUTPUT_ENABLED:
            try:
                self.tts_engine = pyttsx3.init()
                # Configure voice properties
                voices = self.tts_engine.getProperty('voices')
                if voices and len(voices) > 1:
                    self.tts_engine.setProperty('voice', voices[1].id)  # Female voice if available
                self.tts_engine.setProperty('rate', 150)  # Speed of speech
                self.tts_engine.setProperty('volume', 0.8)  # Volume level
                print(f"{Color.GREEN}✅ Text-to-speech engine initialized{Color.END}")
            except Exception as e:
                print(f"{Color.YELLOW}⚠️  Could not initialize TTS: {e}{Color.END}")
                # Don't modify the global flag, just disable for this instance
                self.tts_engine = None
        
        # Initialize speech recognition
        if VOICE_INPUT_ENABLED:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                print(f"{Color.GREEN}✅ Speech recognition initialized{Color.END}")
            except Exception as e:
                print(f"{Color.YELLOW}⚠️  Could not initialize speech recognition: {e}{Color.END}")
                # Don't modify the global flag, just disable for this instance
                self.recognizer = None
    
    def speak(self, text, async_mode=False):
        """Convert text to speech"""
        global VOICE_OUTPUT_ENABLED
        
        if not VOICE_OUTPUT_ENABLED or not self.tts_engine:
            print(f"{Color.YELLOW}💬 [TTS]: {text}{Color.END}")
            return False
            
        try:
            if async_mode:
                # Speak in background thread
                def speak_async():
                    self.is_speaking = True
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.is_speaking = False
                
                thread = threading.Thread(target=speak_async)
                thread.daemon = True
                thread.start()
            else:
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            return True
        except Exception as e:
            print(f"{Color.YELLOW}⚠️  TTS Error: {e}{Color.END}")
            print(f"{Color.YELLOW}💬 [TTS Fallback]: {text}{Color.END}")
            return False
    
    def listen(self, prompt=None, timeout=10):
        """Listen for voice input and convert to text"""
        global VOICE_INPUT_ENABLED
        
        if not VOICE_INPUT_ENABLED or not self.recognizer:
            if prompt:
                print(f"{Color.CYAN}🎤 {prompt} (type your answer): {Color.END}")
            return input().strip().lower()
            
        try:
            with sr.Microphone() as source:
                if prompt:
                    print(f"{Color.CYAN}🎤 {prompt} (speak now)...{Color.END}")
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
                
                # Recognize speech
                text = self.recognizer.recognize_google(audio)
                print(f"{Color.GREEN}🗣️  You said: {text}{Color.END}")
                return text.lower()
                
        except sr.WaitTimeoutError:
            print(f"{Color.YELLOW}⏰ No speech detected within {timeout} seconds{Color.END}")
        except sr.UnknownValueError:
            print(f"{Color.RED}❌ Could not understand audio{Color.END}")
        except sr.RequestError as e:
            print(f"{Color.RED}❌ Speech recognition error: {e}{Color.END}")
        except Exception as e:
            print(f"{Color.RED}❌ Unexpected error: {e}{Color.END}")
        
        # Fallback to text input
        return input(f"{Color.WHITE}📝 Please type your answer: {Color.END}").strip().lower()
    
    def ask_question(self, question, options=None):
        """Ask a question with voice output and get voice/text input"""
        # Speak the question
        self.speak(question)
        
        # Display the question
        print(f"\n{Color.CYAN}💬 {question}{Color.END}")
        
        if options:
            print(f"{Color.BLUE}💡 Options: {', '.join(options)}{Color.END}")
        
        # Get response
        response = self.listen("Your answer")
        return response
    
    def announce_result(self, result):
        """Announce the prediction result using voice"""
        risk_level = result['risk_label']
        probability = result['probability']
        
        announcement = f"Migraine prediction complete. Your risk level is {risk_level} with {probability} percent probability."
        
        # Add recommendations summary
        if result.get('key_recommendations'):
            first_rec = result['key_recommendations'][0]['action']
            announcement += f" Recommended action: {first_rec}"
        
        self.speak(announcement)
    
    def stop(self):
        """Stop any ongoing speech"""
        if self.tts_engine:
            self.tts_engine.stop()

class MigrainePredictorApp:
    def __init__(self):
        self.model = None
        self.voice = VoiceAssistant()
        self.history = []
        self.load_model()
    
    def load_model(self):
        """Load the trained model with colorful output"""
        try:
            print(f"{Color.CYAN}🌀 Loading migraine prediction model...{Color.END}")
            self.model = EnhancedMigrainePredictionModel()
            self.model.load_model('migraine_model.pkl')
            print(f"{Color.GREEN}✅ Model loaded successfully!{Color.END}")
            self.voice.speak("Migraine prediction model loaded and ready")
            return True
        except FileNotFoundError:
            print(f"{Color.RED}❌ Model not found. Please run train_model.py first.{Color.END}")
            self.voice.speak("Model not found. Please train the model first.")
            return False
        except Exception as e:
            print(f"{Color.RED}❌ Error loading model: {e}{Color.END}")
            self.voice.speak("Error loading the prediction model.")
            return False
    
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
        print(f"\n{Color.BOLD}📊 MIGRAINE RISK METER:{Color.END}")
        print("┌" + "─" * 50 + "┐")
        
        # Create visual meter
        meter_length = 50
        filled = int((probability / 100) * meter_length)
        meter_bar = "█" * filled + "░" * (meter_length - filled)
        
        # Color the meter based on risk
        risk_color = self.get_risk_color('green')
        if probability >= 50:
            risk_color = self.get_risk_color('orange')
        if probability >= 70:
            risk_color = self.get_risk_color('red')
        
        print(f"│ {risk_color}{meter_bar}{Color.END} │")
        print(f"│ {Color.WHITE}0%{' ' * 22}{probability}%{' ' * 22}100%{Color.END} │")
        print("└" + "─" * 50 + "┘")
    
    def get_user_input(self):
        """Get input from user for prediction with voice interaction"""
        print(f"\n{Color.CYAN}{'='*50}{Color.END}")
        print(f"{Color.BOLD}🎯 MIGRAINE PREDICTION INPUT{Color.END}")
        print(f"{Color.CYAN}{'='*50}{Color.END}")
        
        # Welcome message
        self.voice.speak("Welcome to the migraine prediction system. Let's assess your current situation.")
        
        inputs = {}
        
        # Stress level with voice
        stress_response = self.voice.ask_question(
            "On a scale of zero to ten, how stressed are you feeling right now?",
            options=["0 for no stress", "10 for extreme stress"]
        )
        try:
            stress = self._parse_number_from_text(stress_response, default=5)
        except:
            stress = 5
        inputs['stress'] = max(0, min(10, stress))
        
        # Mood with voice
        mood_response = self.voice.ask_question(
            "On a scale of zero to ten, how would you rate your current mood? Ten being the best mood.",
            options=["0 for worst mood", "10 for best mood"]
        )
        try:
            mood = self._parse_number_from_text(mood_response, default=5)
        except:
            mood = 5
        inputs['mood'] = max(0, min(10, mood))
        
        # Meal skipping with voice
        meal_questions = [
            ("Did you skip breakfast today?", "skipped_breakfast"),
            ("Did you skip lunch today?", "skipped_lunch"), 
            ("Did you skip dinner today?", "skipped_dinner")
        ]
        
        for question, key in meal_questions:
            response = self.voice.ask_question(question, options=["yes", "no"])
            inputs[key] = 1 if response and 'yes' in response.lower() else 0
        
        # Activity level with voice
        activity_response = self.voice.ask_question(
            "What's your activity level from zero to one hundred? Zero being completely inactive, one hundred being very active.",
            options=["0 to 100"]
        )
        try:
            activity = self._parse_number_from_text(activity_response, default=50)
        except:
            activity = 50
        inputs['activity'] = max(0, min(100, activity))
        
        # Symptoms with voice
        symptoms_response = self.voice.ask_question(
            "Are you experiencing any symptoms like light sensitivity, fatigue, or headache?",
            options=["list symptoms or say none"]
        )
        symptoms = symptoms_response if symptoms_response and 'none' not in symptoms_response.lower() else ""
        
        # Triggers with voice
        triggers_response = self.voice.ask_question(
            "Have you encountered any known triggers today like stress, missed meals, or bright lights?",
            options=["list triggers or say none"]
        )
        triggers = triggers_response if triggers_response and 'none' not in triggers_response.lower() else ""
        
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
        
        # Confirm inputs
        self.voice.speak("Thank you. I have all your information. Now analyzing your migraine risk.")
        
        return user_data, symptoms, triggers, inputs
    
    def _parse_number_from_text(self, text, default=5):
        """Parse number from text response"""
        if not text:
            return default
            
        # Try to find numbers in the response
        words = text.split()
        for word in words:
            if word.isdigit():
                return int(word)
        
        # Check for number words
        number_words = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
        }
        
        for word in words:
            if word in number_words:
                return number_words[word]
        
        return default

    def create_visualization(self, result, user_inputs):
        """Create a visualization of the prediction results"""
        try:
            plt.style.use('default')
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle('Migraine Risk Analysis Dashboard', fontsize=16, fontweight='bold')
            
            # 1. Risk Probability Gauge
            risk_levels = ['Low', 'Moderate', 'High', 'Very High']
            risk_thresholds = [20, 50, 70, 100]
            colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
            
            current_risk = result['probability']
            risk_index = 0
            for i, threshold in enumerate(risk_thresholds):
                if current_risk <= threshold:
                    risk_index = i
                    break
            
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
            plt.close()  # Close the plot to free memory
            print(f"{Color.GREEN}📊 Visualization saved as 'migraine_analysis.png'{Color.END}")
            
        except Exception as e:
            print(f"{Color.YELLOW}⚠️  Could not create visualization: {e}{Color.END}")

    def predict(self):
        """Make prediction with user input and enhanced display"""
        if not self.model:
            print(f"{Color.RED}❌ Model not loaded. Cannot make predictions.{Color.END}")
            return
        
        user_data, symptoms, triggers, user_inputs = self.get_user_input()
        
        print(f"\n{Color.CYAN}🔄 Analyzing your data with AI...{Color.END}")
        self.voice.speak("Analyzing your information with artificial intelligence.")
        
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
            
            # Announce result with voice
            self.voice.announce_result(result)
            
            # Display results with colors
            risk_color = self.get_risk_color(result['risk_band'])
            
            print(f"\n{Color.CYAN}{'='*60}{Color.END}")
            print(f"{Color.BOLD}🎯 MIGRAINE PREDICTION RESULTS{Color.END}")
            print(f"{Color.CYAN}{'='*60}{Color.END}")
            
            # Risk meter
            self.display_risk_meter(result['probability'])
            
            # Main result
            print(f"\n{risk_color}{Color.BOLD}📈 RISK LEVEL: {result['risk_label']} ({result['probability']}% probability){Color.END}")
            print(f"{Color.CYAN}💡 {result['risk_message']}{Color.END}")
            
            # Enhanced display of recommendations
            if result.get('key_recommendations'):
                print(f"\n{Color.BOLD}🛡️  PERSONALIZED RECOMMENDATIONS:{Color.END}")
                for i, rec in enumerate(result['key_recommendations'], 1):
                    urgency_color = {
                        'high': Color.RED,
                        'medium': Color.YELLOW, 
                        'low': Color.GREEN
                    }.get(rec.get('urgency', 'low'), Color.WHITE)
                    
                    print(f"\n{urgency_color}{i}. {rec['action']}{Color.END}")
                    print(f"   {Color.CYAN}💡 Why: {rec['reason']}{Color.END}")
                    
                    # Speak high priority recommendations
                    if rec.get('urgency') == 'high':
                        self.voice.speak(f"High priority recommendation: {rec['action']}")
            
            # Quick actions
            if result.get('quick_actions'):
                print(f"\n{Color.BOLD}⚡ QUICK ACTIONS:{Color.END}")
                for action in result['quick_actions']:
                    print(f"   {Color.GREEN}• {action}{Color.END}")
            
            # Prevention tips
            if result.get('prevention_tips'):
                print(f"\n{Color.BOLD}🎯 PREVENTION TIPS:{Color.END}")
                for tip in result['prevention_tips']:
                    print(f"   {Color.BLUE}• {tip}{Color.END}")
            
            # Risk factors
            if result.get('top_risk_factors'):
                print(f"\n{Color.BOLD}⚠️  IDENTIFIED RISK FACTORS:{Color.END}")
                for factor in result['top_risk_factors']:
                    print(f"   {Color.ORANGE}• {factor}{Color.END}")
            
            print(f"{Color.CYAN}{'='*60}{Color.END}")
            
            # Ask to show visualization
            show_viz = self.voice.ask_question(
                "Would you like to see a detailed visualization of your results?",
                options=["yes", "no"]
            )
            if show_viz and 'yes' in show_viz.lower():
                self.create_visualization(result, user_inputs)
                
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            print(f"{Color.RED}❌ {error_msg}{Color.END}")
            self.voice.speak("Sorry, there was an error processing your prediction.")
            import traceback
            traceback.print_exc()

    def show_history(self):
        """Show prediction history"""
        if not self.history:
            print(f"{Color.YELLOW}No prediction history yet.{Color.END}")
            self.voice.speak("No prediction history available yet.")
            return
            
        print(f"\n{Color.BOLD}📈 PREDICTION HISTORY ({len(self.history)} entries):{Color.END}")
        for i, entry in enumerate(self.history[-5:], 1):  # Show last 5
            result = entry['result']
            risk_color = self.get_risk_color(result['risk_band'])
            print(f"{i}. {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                  f"{risk_color}{result['risk_label']} ({result['probability']}%){Color.END}")
        
        self.voice.speak(f"You have {len(self.history)} predictions in your history.")

    def demo_mode(self):
        """Run a demo with sample data"""
        print(f"\n{Color.MAGENTA}{'='*50}{Color.END}")
        print(f"{Color.BOLD}🎬 DEMO MODE - Sample Prediction{Color.END}")
        print(f"{Color.MAGENTA}{'='*50}{Color.END}")
        
        self.voice.speak("Starting demo mode with sample data.")
        
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
        self.voice.speak("Demo input: High stress, low mood, skipped breakfast, and low activity.")
        
        try:
            result = self.model.predict_with_recommendations(
                sample_data,
                gender='female',
                user_symptoms="light sensitivity, fatigue",
                user_triggers="stress, missed meals"
            )
            
            risk_color = self.get_risk_color(result['risk_band'])
            print(f"\n{risk_color}{Color.BOLD}🎯 DEMO RESULT: {result['risk_label']} ({result['probability']}% probability){Color.END}")
            
            self.voice.speak(f"Demo result: {result['risk_label']} risk with {result['probability']} percent probability.")
            
            # Show visualization for demo
            self.create_visualization(result, user_inputs)
            
        except Exception as e:
            print(f"{Color.RED}❌ Demo error: {e}{Color.END}")
            self.voice.speak("Demo encountered an error.")

    def run(self):
        """Main application loop"""
        if not self.load_model():
            return
        
        # Welcome message
        self.voice.speak("Welcome to the Migraine Prediction System. How can I help you today?")
        
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
            
            choice = self.voice.ask_question(
                "Please choose an option from 1 to 6",
                options=["1 for prediction", "2 for batch", "3 for history", "4 for demo", "5 to retrain", "6 to exit"]
            )
            
            if choice and choice.strip() in ['1', 'one']:
                self.predict()
            elif choice and choice.strip() in ['2', 'two']:
                csv_file = input(f"{Color.CYAN}Enter CSV file path: {Color.END}").strip()
                if os.path.exists(csv_file):
                    self.batch_predict(csv_file)
                else:
                    print(f"{Color.RED}❌ File not found!{Color.END}")
                    self.voice.speak("File not found.")
            elif choice and choice.strip() in ['3', 'three']:
                self.show_history()
            elif choice and choice.strip() in ['4', 'four']:
                self.demo_mode()
            elif choice and choice.strip() in ['5', 'five']:
                print(f"{Color.YELLOW}🔄 Retraining model...{Color.END}")
                self.voice.speak("Retraining the prediction model.")
                os.system('python train_model.py')
                # Reload model after retraining
                self.load_model()
            elif choice and choice.strip() in ['6', 'six', 'exit', 'quit']:
                print(f"{Color.GREEN}👋 Thank you for using Migraine Prediction System!{Color.END}")
                self.voice.speak("Thank you for using the Migraine Prediction System. Goodbye!")
                break
            else:
                print(f"{Color.RED}❌ Invalid choice. Please try again.{Color.END}")
                self.voice.speak("Invalid choice. Please try again.")

    def batch_predict(self, csv_file):
        """Make predictions from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} records from {csv_file}")
            self.voice.speak(f"Loaded {len(df)} records for batch prediction.")
            
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
            self.voice.speak("Error during batch prediction.")

if __name__ == "__main__":
    app = MigrainePredictorApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}👋 Session ended by user.{Color.END}")
        app.voice.speak("Session ended.")
    finally:
        app.voice.stop()