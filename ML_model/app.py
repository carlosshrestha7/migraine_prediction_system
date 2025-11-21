# app.py
"""
Enhanced CLI for Migraine Prediction System with Complete Voice Interaction
"""

import pandas as pd
from enhanced_migraine_model import EnhancedMigrainePredictionModel, format_recommendations_for_display
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import random
import threading
import time

# Global voice flags
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
        
        # Initialize speech recognition
        if VOICE_INPUT_ENABLED:
            try:
                self.recognizer = sr.Recognizer()
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                print(f"{Color.GREEN}✅ Speech recognition initialized{Color.END}")
            except Exception as e:
                print(f"{Color.YELLOW}⚠️  Could not initialize speech recognition: {e}{Color.END}")
    
    def speak(self, text, async_mode=False, wait_until_done=False):
        """Convert text to speech"""
        if not VOICE_OUTPUT_ENABLED or not self.tts_engine:
            print(f"{Color.YELLOW}💬 [TTS]: {text}{Color.END}")
            return False
            
        try:
            if async_mode and not wait_until_done:
                # Speak in background thread
                def speak_async():
                    self.is_speaking = True
                    self.tts_engine.say(text)
                    self.tts_engine.runAndWait()
                    self.is_speaking = False
                
                thread = threading.Thread(target=speak_async)
                thread.daemon = True
                thread.start()
                return True
            else:
                # Block until speaking is done
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
                return True
        except Exception as e:
            print(f"{Color.YELLOW}⚠️  TTS Error: {e}{Color.END}")
            print(f"{Color.YELLOW}💬 [TTS Fallback]: {text}{Color.END}")
            return False
    
    def wait_for_speech_completion(self, timeout=10):
        """Wait for current speech to complete"""
        start_time = time.time()
        while self.is_speaking and (time.time() - start_time) < timeout:
            time.sleep(0.1)
    
    def listen(self, prompt=None, timeout=8):
        """Listen for voice input and convert to text"""
        if not VOICE_INPUT_ENABLED or not self.recognizer:
            if prompt:
                print(f"{Color.CYAN}🎤 {prompt} (type your answer): {Color.END}")
            return None
            
        try:
            with sr.Microphone() as source:
                if prompt:
                    print(f"{Color.CYAN}🎤 {prompt} (speak now)...{Color.END}")
                
                # Wait a moment after question is spoken
                time.sleep(1)
                
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                
                # Listen for audio with longer timeout
                print(f"{Color.BLUE}🎧 Listening for {timeout} seconds...{Color.END}")
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=10)
                
                # Recognize speech
                print(f"{Color.BLUE}🔄 Processing your speech...{Color.END}")
                text = self.recognizer.recognize_google(audio)
                print(f"{Color.GREEN}🗣️  You said: {text}{Color.END}")
                return text.lower()
                
        except sr.WaitTimeoutError:
            print(f"{Color.YELLOW}⏰ No speech detected within {timeout} seconds{Color.END}")
        except sr.UnknownValueError:
            print(f"{Color.RED}❌ Could not understand audio. Please try again.{Color.END}")
        except sr.RequestError as e:
            print(f"{Color.RED}❌ Speech recognition error: {e}{Color.END}")
        except Exception as e:
            print(f"{Color.RED}❌ Unexpected error: {e}{Color.END}")
        
        return None
    
    def ask_question(self, question, options=None, listen_timeout=8):
        """Ask a question with voice output and get voice/text input"""
        # Speak the question and wait for it to complete
        print(f"\n{Color.CYAN}{'='*60}{Color.END}")
        self.speak(question, wait_until_done=True)
        
        # Display the question
        print(f"{Color.CYAN}💬 {question}{Color.END}")
        
        if options:
            print(f"{Color.BLUE}💡 Options: {', '.join(options)}{Color.END}")
            # Also speak options with a brief pause
            time.sleep(0.5)
            options_text = f"Options are: {', '.join(options)}"
            self.speak(options_text, wait_until_done=True)
        
        # Brief pause after speaking the question
        time.sleep(0.5)
        
        # Get response with voice
        if VOICE_INPUT_ENABLED:
            response = self.listen("Your answer", timeout=listen_timeout)
            if response is not None:
                return response
            else:
                print(f"{Color.YELLOW}🔇 Voice input failed. Switching to text input.{Color.END}")
        
        # Fallback to text input
        user_input = input(f"{Color.WHITE}📝 Your answer: {Color.END}").strip()
        return user_input
    
    def ask_numeric_question(self, question, min_val=0, max_val=10, default=5):
        """Ask a numeric question with voice and parse the response"""
        response = self.ask_question(
            f"{question} Please say a number between {min_val} and {max_val}.",
            options=[f"{min_val} to {max_val}"],
            listen_timeout=10
        )
        
        if response:
            try:
                # Try to extract number from speech
                numbers = [int(s) for s in response.split() if s.isdigit()]
                if numbers:
                    number = numbers[0]
                    if min_val <= number <= max_val:
                        return number
                
                # Try word-to-number conversion
                word_to_number = {
                    'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
                    'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10
                }
                
                for word, num in word_to_number.items():
                    if word in response:
                        if min_val <= num <= max_val:
                            return num
                
                print(f"{Color.YELLOW}⚠️  Could not understand number. Using default: {default}{Color.END}")
                
            except Exception as e:
                print(f"{Color.YELLOW}⚠️  Error parsing number: {e}. Using default: {default}{Color.END}")
        
        return default
    
    def ask_yes_no_question(self, question):
        """Ask a yes/no question with voice"""
        response = self.ask_question(
            question,
            options=["yes", "no"],
            listen_timeout=6
        )
        
        if response:
            if any(word in response for word in ['yes', 'yeah', 'yep', 'sure', 'okay']):
                return True
            elif any(word in response for word in ['no', 'nope', 'nah', 'negative']):
                return False
        
        # Fallback: ask for text input
        print(f"{Color.YELLOW}🔇 Could not understand voice response.{Color.END}")
        text_response = input(f"{Color.WHITE}📝 Please type 'yes' or 'no': {Color.END}").strip().lower()
        return text_response in ['yes', 'y', '1']
    
    def announce_result(self, result):
        """Announce the prediction result using voice"""
        risk_level = result['risk_label']
        probability = result['probability']
        
        announcement = f"Migraine prediction complete. Your risk level is {risk_level} with {probability} percent probability."
        
        # Add risk-specific message
        if result['risk_band'] == 'red':
            announcement += " This is a very high risk. Please take immediate action."
        elif result['risk_band'] == 'orange':
            announcement += " This is high risk. Consider preventive measures."
        elif result['risk_band'] == 'yellow':
            announcement += " This is moderate risk. Stay vigilant."
        else:
            announcement += " This is low risk. Continue with your prevention routine."
        
        self.speak(announcement, wait_until_done=True)
    
    def stop(self):
        """Stop any ongoing speech"""
        if self.tts_engine and VOICE_OUTPUT_ENABLED:
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
            self.voice.speak("Migraine prediction model loaded and ready.", wait_until_done=True)
        except FileNotFoundError:
            print(f"{Color.RED}❌ Model not found. Please run train_model.py first.{Color.END}")
            self.voice.speak("Model not found. Please train the model first.", wait_until_done=True)
            return False
        except Exception as e:
            print(f"{Color.RED}❌ Error loading model: {e}{Color.END}")
            self.voice.speak("Error loading the prediction model.", wait_until_done=True)
            return False
        return True
    
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
        meter_bar = f"{Color.RED}{'█' * filled}{Color.END}{Color.WHITE}{'░' * (meter_length - filled)}{Color.END}"
        
        print(f"│ {meter_bar} │")
        print(f"│ {Color.WHITE}0%{' ' * 22}{probability}%{' ' * 22}100%{Color.END} │")
        print("└" + "─" * 50 + "┘")
    
    def get_user_input(self):
        """Get input from user for prediction with complete voice interaction"""
        print(f"\n{Color.CYAN}{'='*50}{Color.END}")
        print(f"{Color.BOLD}🎯 MIGRAINE PREDICTION INPUT{Color.END}")
        print(f"{Color.CYAN}{'='*50}{Color.END}")
        
        # Welcome message with voice
        welcome_msg = "Welcome to the migraine prediction system. I will ask you a few questions about your current state. Please answer each question after you hear the beep."
        self.voice.speak(welcome_msg, wait_until_done=True)
        time.sleep(1)
        
        inputs = {}
        
        # Stress level with voice
        print(f"\n{Color.MAGENTA}1. STRESS ASSESSMENT{Color.END}")
        inputs['stress'] = self.voice.ask_numeric_question(
            "On a scale of zero to ten, how stressed are you feeling right now?",
            min_val=0, max_val=10, default=5
        )
        print(f"{Color.GREEN}✅ Stress level: {inputs['stress']}/10{Color.END}")
        
        # Mood with voice
        print(f"\n{Color.MAGENTA}2. MOOD ASSESSMENT{Color.END}")
        inputs['mood'] = self.voice.ask_numeric_question(
            "On a scale of zero to ten, how would you rate your current mood? Ten being the best mood possible.",
            min_val=0, max_val=10, default=5
        )
        print(f"{Color.GREEN}✅ Mood level: {inputs['mood']}/10{Color.END}")
        
        # Meal skipping with voice
        print(f"\n{Color.MAGENTA}3. MEAL STATUS{Color.END}")
        meal_questions = [
            ("Did you skip breakfast today?", "skipped_breakfast"),
            ("Did you skip lunch today?", "skipped_lunch"), 
            ("Did you skip dinner today?", "skipped_dinner")
        ]
        
        for question, key in meal_questions:
            inputs[key] = self.voice.ask_yes_no_question(question)
            status = "skipped" if inputs[key] else "had"
            print(f"{Color.GREEN}✅ {question.split('?')[0]}: {status}{Color.END}")
        
        # Activity level with voice
        print(f"\n{Color.MAGENTA}4. ACTIVITY LEVEL{Color.END}")
        inputs['activity'] = self.voice.ask_numeric_question(
            "What is your current activity level from zero to one hundred? Zero means completely inactive, one hundred means very active.",
            min_val=0, max_val=100, default=50
        )
        print(f"{Color.GREEN}✅ Activity level: {inputs['activity']}/100{Color.END}")
        
        # Symptoms with voice
        print(f"\n{Color.MAGENTA}5. SYMPTOMS CHECK{Color.END}")
        symptoms_response = self.voice.ask_question(
            "Are you experiencing any symptoms right now? For example: light sensitivity, fatigue, headache, nausea, or sound sensitivity. If none, just say 'none'.",
            options=["list symptoms", "say 'none'"],
            listen_timeout=12
        )
        symptoms = symptoms_response if symptoms_response and 'none' not in symptoms_response.lower() else ""
        if symptoms:
            print(f"{Color.GREEN}✅ Symptoms reported: {symptoms}{Color.END}")
        else:
            print(f"{Color.GREEN}✅ No symptoms reported{Color.END}")
        
        # Triggers with voice
        print(f"\n{Color.MAGENTA}6. TRIGGERS CHECK{Color.END}")
        triggers_response = self.voice.ask_question(
            "Have you encountered any known triggers today? For example: stress, missed meals, bright lights, loud noises, or weather changes. If none, just say 'none'.",
            options=["list triggers", "say 'none'"],
            listen_timeout=12
        )
        triggers = triggers_response if triggers_response and 'none' not in triggers_response.lower() else ""
        if triggers:
            print(f"{Color.GREEN}✅ Triggers reported: {triggers}{Color.END}")
        else:
            print(f"{Color.GREEN}✅ No triggers reported{Color.END}")
        
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
        
        # Confirm inputs with voice
        confirmation_msg = f"Thank you. I have recorded your stress level as {inputs['stress']} out of 10, your mood as {inputs['mood']} out of 10, and your activity level as {inputs['activity']} out of 100. Now analyzing your migraine risk."
        self.voice.speak(confirmation_msg, wait_until_done=True)
        
        return user_data, symptoms, triggers, inputs
    
    def create_visualization(self, result, user_inputs):
        """Create a visualization of the prediction results"""
        try:
            # Check if matplotlib can display
            import matplotlib
            matplotlib.use('TkAgg')  # Use Tkinter backend
            
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
            
            # Try to show plot, but continue if it fails
            try:
                plt.show()
            except:
                print(f"{Color.YELLOW}⚠️  Could not display plot, but image was saved.{Color.END}")
            
        except Exception as e:
            print(f"{Color.YELLOW}⚠️  Could not create visualization: {e}{Color.END}")
    
    def predict(self):
        """Make prediction with user input and enhanced display"""
        if not self.model:
            print(f"{Color.RED}❌ Model not loaded. Cannot make predictions.{Color.END}")
            return
        
        user_data, symptoms, triggers, user_inputs = self.get_user_input()
        
        print(f"\n{Color.CYAN}🔄 Analyzing your data with AI...{Color.END}")
        self.voice.speak("Analyzing your information with artificial intelligence. Please wait a moment.", wait_until_done=True)
        time.sleep(2)  # Simulate processing time
        
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
                
                # Speak high priority recommendations
                if rec.get('urgency') == 'high':
                    self.voice.speak(f"High priority recommendation: {rec['action']}", wait_until_done=True)
            
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
            
            # Ask to show visualization with voice
            show_viz_response = self.voice.ask_yes_no_question(
                "Would you like to see a detailed visualization of your results?"
            )
            if show_viz_response:
                self.voice.speak("Generating your visualization now.", wait_until_done=True)
                self.create_visualization(result, user_inputs)
            else:
                self.voice.speak("Okay, returning to main menu.", wait_until_done=True)
                
        except Exception as e:
            error_msg = f"Prediction error: {e}"
            print(f"{Color.RED}❌ {error_msg}{Color.END}")
            self.voice.speak("Sorry, there was an error processing your prediction. Please try again.", wait_until_done=True)
            import traceback
            traceback.print_exc()
    
    def show_history(self):
        """Show prediction history"""
        if not self.history:
            print(f"{Color.YELLOW}No prediction history yet.{Color.END}")
            self.voice.speak("No prediction history available yet.", wait_until_done=True)
            return
            
        print(f"\n{Color.BOLD}📈 PREDICTION HISTORY ({len(self.history)} entries):{Color.END}")
        for i, entry in enumerate(self.history[-5:], 1):  # Show last 5
            result = entry['result']
            risk_color = self.get_risk_color(result['risk_band'])
            print(f"{i}. {entry['timestamp'].strftime('%Y-%m-%d %H:%M')} - "
                  f"{risk_color}{result['risk_label']} ({result['probability']}%){Color.END}")
        
        self.voice.speak(f"You have {len(self.history)} predictions in your history.", wait_until_done=True)
    
    def demo_mode(self):
        """Run a demo with sample data"""
        print(f"\n{Color.MAGENTA}{'='*50}{Color.END}")
        print(f"{Color.BOLD}🎬 DEMO MODE - Sample Prediction{Color.END}")
        print(f"{Color.MAGENTA}{'='*50}{Color.END}")
        
        self.voice.speak("Starting demo mode with sample data.", wait_until_done=True)
        
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
        self.voice.speak("Demo input: High stress, low mood, skipped breakfast, and low activity.", wait_until_done=True)
        
        try:
            result = self.model.predict_with_recommendations(
                sample_data,
                gender='female',
                user_symptoms="light sensitivity, fatigue",
                user_triggers="stress, missed meals"
            )
            
            risk_color = self.get_risk_color(result['risk_band'])
            print(f"\n{risk_color}{Color.BOLD}🎯 DEMO RESULT: {result['risk_label']} ({result['probability']}% probability){Color.END}")
            
            self.voice.speak(f"Demo result: {result['risk_label']} risk with {result['probability']} percent probability.", wait_until_done=True)
            
            # Show visualization for demo
            self.create_visualization(result, user_inputs)
            
        except Exception as e:
            print(f"{Color.RED}❌ Demo error: {e}{Color.END}")
            self.voice.speak("Demo encountered an error.", wait_until_done=True)
    
    def run(self):
        """Main application loop"""
        if not self.load_model():
            return
        
        # Welcome message
        welcome_msg = "Welcome to the Migraine Prediction System. I can help you assess your migraine risk and provide personalized recommendations. How can I help you today?"
        self.voice.speak(welcome_msg, wait_until_done=True)
        
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
            
            choice = self.voice.ask_numeric_question(
                "Please choose an option from 1 to 6",
                min_val=1, max_val=6, default=1
            )
            
            if choice == 1:
                self.predict()
            elif choice == 2:
                csv_file = input(f"{Color.CYAN}Enter CSV file path: {Color.END}").strip()
                if os.path.exists(csv_file):
                    self.batch_predict(csv_file)
                else:
                    print(f"{Color.RED}❌ File not found!{Color.END}")
                    self.voice.speak("File not found.", wait_until_done=True)
            elif choice == 3:
                self.show_history()
            elif choice == 4:
                self.demo_mode()
            elif choice == 5:
                print(f"{Color.YELLOW}🔄 Retraining model...{Color.END}")
                self.voice.speak("Retraining the prediction model. This may take a few moments.", wait_until_done=True)
                os.system('python train_model.py')
            elif choice == 6:
                print(f"{Color.GREEN}👋 Thank you for using Migraine Prediction System!{Color.END}")
                self.voice.speak("Thank you for using the Migraine Prediction System. Goodbye and take care!", wait_until_done=True)
                break
            else:
                print(f"{Color.RED}❌ Invalid choice. Please try again.{Color.END}")
                self.voice.speak("Invalid choice. Please try again.", wait_until_done=True)

    def batch_predict(self, csv_file):
        """Make predictions from CSV file"""
        try:
            df = pd.read_csv(csv_file)
            print(f"📊 Loaded {len(df)} records from {csv_file}")
            self.voice.speak(f"Loaded {len(df)} records for batch prediction.", wait_until_done=True)
            
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
            self.voice.speak("Error during batch prediction.", wait_until_done=True)

if __name__ == "__main__":
    app = MigrainePredictorApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}👋 Session ended by user.{Color.END}")
        app.voice.speak("Session ended.", wait_until_done=True)
    finally:
        app.voice.stop()