"""
app.py - GUI Migraine Prediction System with robust TTS + STT timing fixes

Features:
- Tkinter GUI for guided question flow
- VoiceAssistant with fixed timing (prevents mic from listening to TTS)
- Listen runs in background threads; GUI remains responsive
- Users see the question and hear it; mic activates only after speech is finished
- Fallback text input editable before confirming
"""

import os
import sys
import threading
import time
from datetime import datetime
import queue
import random

import pandas as pd
import matplotlib.pyplot as plt

# ---- optional voice packages ----
VOICE_INPUT_ENABLED = False
VOICE_OUTPUT_ENABLED = False
try:
    import speech_recognition as sr
    VOICE_INPUT_ENABLED = True
except Exception:
    VOICE_INPUT_ENABLED = False
    print("⚠️  SpeechRecognition not available. Voice input disabled.")

try:
    import pyttsx3
    VOICE_OUTPUT_ENABLED = True
except Exception:
    VOICE_OUTPUT_ENABLED = False
    print("⚠️  pyttsx3 not available. Voice output disabled.")

# Your model import (unchanged)
try:
    from enhanced_migraine_model import EnhancedMigrainePredictionModel
except Exception:
    # Keep import but allow file to run even without model during UI/voice testing
    EnhancedMigrainePredictionModel = None

# Tkinter imports
try:
    import tkinter as tk
    from tkinter import ttk, messagebox, filedialog
except Exception as e:
    print("Tkinter not available. GUI cannot start.")
    raise

# ANSI colors for console logs (not for GUI)
class Color:
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'

# ---------------- VoiceAssistant (fixed timing) ----------------
class VoiceAssistant:
    def __init__(self):
        self.tts_engine = None
        self.recognizer = None
        self.is_speaking = False
        self._listen_thread = None
        self._stop_listen_flag = threading.Event()

        if VOICE_OUTPUT_ENABLED:
            try:
                self.tts_engine = pyttsx3.init()
                voices = self.tts_engine.getProperty('voices')
                if voices and len(voices) > 1:
                    # try pick a different voice if available
                    self.tts_engine.setProperty('voice', voices[1].id)
                self.tts_engine.setProperty('rate', 150)
                self.tts_engine.setProperty('volume', 0.9)
                print(f"{Color.GREEN}✅ TTS initialized{Color.END}")
            except Exception as e:
                print(f"{Color.YELLOW}⚠️ TTS init failed: {e}{Color.END}")
                self.tts_engine = None

        if VOICE_INPUT_ENABLED:
            try:
                self.recognizer = sr.Recognizer()
                # tuned defaults
                self.recognizer.energy_threshold = 300
                self.recognizer.dynamic_energy_threshold = True
                self.recognizer.pause_threshold = 0.6
                print(f"{Color.GREEN}✅ Speech recognition ready{Color.END}")
            except Exception as e:
                print(f"{Color.YELLOW}⚠️ SR init failed: {e}{Color.END}")
                self.recognizer = None

    # Blocking speak that ensures the audio hardware is finished (buffer flushed) before returning.
    def speak_blocking(self, text, on_start=None, on_end=None):
        """
        Speak text, blocking until speech is fully done.
        Calls on_start() just before speaking and on_end() after speech finished.
        """
        print(f"[TTS] {text}")
        if on_start:
            try:
                on_start()
            except Exception:
                pass

        if not VOICE_OUTPUT_ENABLED or not self.tts_engine:
            # just wait a little so GUI timing remains consistent
            time.sleep(0.5)
            if on_end:
                try:
                    on_end()
                except Exception:
                    pass
            return False

        try:
            # Mark speaking
            self.is_speaking = True
            # say and wait
            self.tts_engine.say(text)
            self.tts_engine.runAndWait()
            # small safety buffer to allow audio hardware to finish
            time.sleep(0.55)
            self.is_speaking = False
            if on_end:
                try:
                    on_end()
                except Exception:
                    pass
            return True
        except Exception as e:
            print(f"{Color.YELLOW}⚠️ TTS error: {e}{Color.END}")
            self.is_speaking = False
            if on_end:
                try:
                    on_end()
                except Exception:
                    pass
            return False

    def speak_async(self, text, on_start=None, on_end=None):
        """Speak in a background thread (non-blocking for caller)."""
        t = threading.Thread(target=self.speak_blocking, args=(text, on_start, on_end), daemon=True)
        t.start()
        return t

    def stop(self):
        """Stop TTS if possible."""
        try:
            if self.tts_engine:
                self.tts_engine.stop()
        except Exception:
            pass
        self.is_speaking = False
        # signal any listening threads to stop
        self._stop_listen_flag.set()

    def _safe_listen_worker(self, callback, timeout, phrase_time_limit):
        """
        Worker that waits until TTS finished, sleeps a short safety delay,
        then captures audio and calls callback(transcribed_text or None).
        """
        # Wait until tts is fully finished
        # This prevents the mic from capturing the TTS voice
        while self.is_speaking:
            time.sleep(0.05)

        # Extra safety buffer (makes it very unlikely mic hears TTS)
        time.sleep(0.55)

        if not VOICE_INPUT_ENABLED or not self.recognizer:
            callback(None, "voice_disabled")
            return

        try:
            with sr.Microphone() as source:
                # brief ambient adjustment (short)
                self.recognizer.adjust_for_ambient_noise(source, duration=0.7)
                # Check stop flag before starting listen
                if self._stop_listen_flag.is_set():
                    callback(None, "stopped")
                    return

                print("[STT] Listening (you may speak now)...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )
                if self._stop_listen_flag.is_set():
                    callback(None, "stopped")
                    return

                print("[STT] Processing...")
                text = self.recognizer.recognize_google(audio)
                callback(text, None)
                return

        except sr.WaitTimeoutError:
            callback(None, "timeout")
            return
        except sr.UnknownValueError:
            callback(None, "unknown")
            return
        except sr.RequestError as e:
            callback(None, f"request_error:{e}")
            return
        except Exception as e:
            callback(None, f"error:{e}")
            return

    def listen_async(self, callback, timeout=8, phrase_time_limit=8):
        """
        Start listening in the background. callback(transcribed_text, error) will be called.
        error is None if success, otherwise one of 'timeout','unknown','request_error:...','voice_disabled',etc.
        """
        # Reset stop flag
        self._stop_listen_flag.clear()
        t = threading.Thread(target=self._safe_listen_worker, args=(callback, timeout, phrase_time_limit), daemon=True)
        t.start()
        self._listen_thread = t
        return t

# ---------------- GUI Application ----------------
class MigraineGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Migraine Prediction System")
        self.geometry("920x640")
        self.resizable(False, False)

        # Model
        self.model = None
        if EnhancedMigrainePredictionModel:
            try:
                self.model = EnhancedMigrainePredictionModel()
                # attempt to load model if available; silence errors for dev
                try:
                    self.model.load_model('migraine_model.pkl')
                except Exception:
                    pass
            except Exception:
                self.model = None

        # Voice assistant
        self.voice = VoiceAssistant()

        # History
        self.history = []

        # question flow
        self.questions = [
            {'id': 'stress', 'type': 'numeric', 'label': 'On a scale of 0–10, how stressed are you right now?', 'min':0,'max':10,'default':5},
            {'id': 'mood', 'type': 'numeric', 'label': 'On a scale of 0–10, how would you rate your current mood?', 'min':0,'max':10,'default':5},
            {'id': 'skipped_breakfast', 'type': 'yesno', 'label': 'Did you skip breakfast today?'},
            {'id': 'skipped_lunch', 'type': 'yesno', 'label': 'Did you skip lunch today?'},
            {'id': 'skipped_dinner', 'type': 'yesno', 'label': 'Did you skip dinner today?'},
            {'id': 'activity', 'type': 'numeric', 'label': 'What is your current activity level (0–100)?', 'min':0,'max':100,'default':50},
            {'id': 'symptoms', 'type': 'open', 'label': 'Are you experiencing any symptoms right now? (e.g. light sensitivity, nausea)'},
            {'id': 'triggers', 'type': 'open', 'label': 'Have you encountered any known triggers today? (e.g. stress, missed meals)'}
        ]
        self.current_index = 0
        self.responses = {}

        # GUI elements
        self._build_ui()

        # Start with idle message and load model status
        self._update_status("Ready. Press Start to begin questionnaire.")
        if self.model:
            self._update_status("Model loaded.", append=True)
        else:
            self._update_status("Model not loaded — you can still test voice/UI.", append=True)

    def _build_ui(self):
        # Top frame: header and status
        header = ttk.Frame(self, padding=(12,12))
        header.pack(fill=tk.X)

        ttk.Label(header, text="Migraine Prediction System", font=("Segoe UI", 18, "bold")).pack(side=tk.LEFT)
        self.status_label = ttk.Label(header, text="", font=("Segoe UI", 10))
        self.status_label.pack(side=tk.RIGHT)

        # Content frame
        content = ttk.Frame(self, padding=(12,8))
        content.pack(fill=tk.BOTH, expand=True)

        # Left: question pane
        left = ttk.Frame(content)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,8))

        # Question card
        card = ttk.LabelFrame(left, text="Question", padding=(12,12))
        card.pack(fill=tk.BOTH, expand=True)

        self.question_text = tk.Text(card, height=4, wrap="word", font=("Segoe UI", 12))
        self.question_text.pack(fill=tk.X, pady=(0,8))
        self.question_text.configure(state='disabled')

        # Recognized / input field
        self.answer_var = tk.StringVar()
        answer_frame = ttk.Frame(card)
        answer_frame.pack(fill=tk.X, pady=(4,4))

        ttk.Label(answer_frame, text="Your answer:", font=("Segoe UI", 10)).pack(anchor=tk.W)
        self.answer_entry = ttk.Entry(answer_frame, textvariable=self.answer_var, font=("Segoe UI", 12))
        self.answer_entry.pack(fill=tk.X, pady=(4,6))

        # Controls for question
        controls = ttk.Frame(card)
        controls.pack(fill=tk.X)

        self.speak_btn = ttk.Button(controls, text="🔊 Speak Question", command=self._on_speak_question)
        self.speak_btn.pack(side=tk.LEFT, padx=(0,6))

        self.listen_btn = ttk.Button(controls, text="🎧 Auto Listen", command=self._on_listen_now)
        self.listen_btn.pack(side=tk.LEFT, padx=(0,6))

        self.confirm_btn = ttk.Button(controls, text="✅ Confirm Answer", command=self._confirm_answer)
        self.confirm_btn.pack(side=tk.RIGHT, padx=(6,0))

        self.next_btn = ttk.Button(controls, text="➡ Next", command=self._next_question)
        self.next_btn.pack(side=tk.RIGHT, padx=(6,0))

        # Right: actions and preview
        right = ttk.Frame(content, width=300)
        right.pack(side=tk.RIGHT, fill=tk.Y)

        actions = ttk.LabelFrame(right, text="Actions", padding=(8,8))
        actions.pack(fill=tk.X, pady=(0,8))

        ttk.Button(actions, text="Start Questionnaire", command=self.start_questionnaire).pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Make Prediction", command=self._on_predict).pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Demo Mode", command=self._on_demo).pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Batch Predict (CSV)", command=self._on_batch_predict).pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Show History", command=self._show_history).pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="Exit", command=self._on_exit).pack(fill=tk.X, pady=4)

        # Visual indicator area
        indicator = ttk.LabelFrame(right, text="Voice Status", padding=(8,8))
        indicator.pack(fill=tk.X, pady=(8,0))

        self.tts_status = ttk.Label(indicator, text="TTS: idle")
        self.tts_status.pack(anchor=tk.W, pady=(0,4))
        self.stt_status = ttk.Label(indicator, text="STT: idle")
        self.stt_status.pack(anchor=tk.W)

        # Bottom: log / result area
        bottom = ttk.LabelFrame(self, text="Log / Results", padding=(8,8))
        bottom.pack(fill=tk.BOTH, expand=True, padx=12, pady=(6,12))
        self.log_text = tk.Text(bottom, height=8, state='disabled', wrap='word', font=("Segoe UI", 10))
        self.log_text.pack(fill=tk.BOTH, expand=True)

    # ---------------- UI helper functions ----------------
    def _update_status(self, text, append=False):
        if append:
            self.status_label.config(text=self.status_label.cget("text") + " | " + text)
        else:
            self.status_label.config(text=text)
        self._log(text)

    def _log(self, message):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.log_text.configure(state='normal')
        self.log_text.insert("end", f"[{ts}] {message}\n")
        self.log_text.see("end")
        self.log_text.configure(state='disabled')

    def _set_tts_indicator(self, text):
        self.tts_status.config(text=f"TTS: {text}")

    def _set_stt_indicator(self, text):
        self.stt_status.config(text=f"STT: {text}")

    # ---------------- Questionnaire flow ----------------
    def start_questionnaire(self):
        self.current_index = 0
        self.responses = {}
        self._show_current_question()
        self._update_status("Questionnaire started.")

    def _show_current_question(self):
        if self.current_index < 0 or self.current_index >= len(self.questions):
            self.question_text.configure(state='normal')
            self.question_text.delete("1.0", tk.END)
            self.question_text.insert(tk.END, "Questionnaire complete. Press Make Prediction to analyze results.")
            self.question_text.configure(state='disabled')
            return

        q = self.questions[self.current_index]
        self.question_text.configure(state='normal')
        self.question_text.delete("1.0", tk.END)
        self.question_text.insert(tk.END, q['label'])
        self.question_text.configure(state='disabled')

        # Prepopulate entry with previous answer if exists or default
        existing = self.responses.get(q['id'])
        if existing is not None:
            self.answer_var.set(str(existing))
        else:
            if q['type'] == 'numeric':
                self.answer_var.set(str(q.get('default', '')))
            else:
                self.answer_var.set("")

        # Auto-speak question
        self._on_speak_question()

    def _on_speak_question(self):
        q = self.questions[self.current_index]
        text = q['label']
        # update indicators
        self._set_tts_indicator("speaking")
        self._set_stt_indicator("idle")

        # When TTS starts/ends we update indicators via callbacks
        def on_start():
            self._set_tts_indicator("speaking")
            # disable listening while speaking
            self.listen_btn.configure(state=tk.DISABLED)
            self.speak_btn.configure(state=tk.DISABLED)
            self.confirm_btn.configure(state=tk.DISABLED)
            self.next_btn.configure(state=tk.DISABLED)

        def on_end():
            self._set_tts_indicator("idle")
            # enable listening and controls
            self.listen_btn.configure(state=tk.NORMAL)
            self.speak_btn.configure(state=tk.NORMAL)
            self.confirm_btn.configure(state=tk.NORMAL)
            self.next_btn.configure(state=tk.NORMAL)
            # after a small buffer, start listening automatically
            # we use voice.listen_async which already waits for TTS to finish
            # start auto-listen
            # Use a short delay so GUI updates
            self.after(200, lambda: self._auto_listen_after_tts())

        # run speak in background so GUI isn't blocked
        self.voice.speak_async(text, on_start=on_start, on_end=on_end)

    def _auto_listen_after_tts(self):
        # Auto-trigger listen (same as pressing listen button)
        self._on_listen_now()

    def _on_listen_now(self):
        # disable controls while listening
        self.listen_btn.configure(state=tk.DISABLED)
        self.speak_btn.configure(state=tk.DISABLED)
        self.confirm_btn.configure(state=tk.DISABLED)
        self.next_btn.configure(state=tk.DISABLED)
        self._set_stt_indicator("listening")

        def callback(result_text, error):
            # called in background thread; switch to main thread to update UI
            def _finish():
                if error is None:
                    self._set_stt_indicator("recognized")
                    self._log(f"Recognized: {result_text}")
                    self.answer_var.set(result_text)
                else:
                    # friendly user messages
                    if error == "timeout":
                        self._set_stt_indicator("timeout")
                        self._log("No speech detected (timeout).")
                    elif error == "unknown":
                        self._set_stt_indicator("unrecognized")
                        self._log("Speech not understood.")
                    elif error.startswith("request_error"):
                        self._set_stt_indicator("error")
                        self._log(f"STT request error: {error}")
                    elif error == "voice_disabled":
                        self._set_stt_indicator("disabled")
                        self._log("Voice input not available.")
                    elif error == "stopped":
                        self._set_stt_indicator("stopped")
                        self._log("Listening aborted.")
                    else:
                        self._set_stt_indicator("error")
                        self._log(f"Listening error: {error}")
                # re-enable controls
                self.listen_btn.configure(state=tk.NORMAL)
                self.speak_btn.configure(state=tk.NORMAL)
                self.confirm_btn.configure(state=tk.NORMAL)
                self.next_btn.configure(state=tk.NORMAL)

            self.after(10, _finish)

        # start listening async with a callback
        self.voice.listen_async(callback, timeout=8, phrase_time_limit=8)

    def _confirm_answer(self):
        q = self.questions[self.current_index]
        val = self.answer_var.get().strip()
        if q['type'] == 'numeric':
            # try to parse numeric, otherwise warn
            try:
                num = float(val)
                # apply bounds
                minv = q.get('min', None)
                maxv = q.get('max', None)
                if minv is not None and num < minv:
                    num = minv
                if maxv is not None and num > maxv:
                    num = maxv
                # store as int if integer-like
                if num.is_integer():
                    num = int(num)
                self.responses[q['id']] = num
                self._log(f"Answer recorded: {q['id']} = {num}")
            except Exception:
                messagebox.showwarning("Invalid input", "Please enter a numeric value.")
                return
        elif q['type'] == 'yesno':
            # interpret yes/no words or 1/0
            v = val.lower()
            if v in ['yes', 'y', 'true', '1', 'skip', 'skipped']:
                self.responses[q['id']] = 1
            elif v in ['no', 'n', 'false', '0', 'had']:
                self.responses[q['id']] = 0
            else:
                # try int parse
                try:
                    iv = int(val)
                    self.responses[q['id']] = 1 if iv != 0 else 0
                except Exception:
                    messagebox.showwarning("Invalid input", "Please answer 'yes' or 'no' (or type 1/0).")
                    return
            self._log(f"Answer recorded: {q['id']} = {self.responses[q['id']]}")
        else:  # open text
            self.responses[q['id']] = val
            self._log(f"Answer recorded: {q['id']} = {val}")

        # After confirming, automatically move to next question
        self._next_question()

    def _next_question(self):
        self.current_index += 1
        if self.current_index >= len(self.questions):
            self.question_text.configure(state='normal')
            self.question_text.delete("1.0", tk.END)
            self.question_text.insert(tk.END, "Questionnaire finished. Press 'Make Prediction' to get results.")
            self.question_text.configure(state='disabled')
            self._update_status("Questionnaire complete.")
            # re-enable controls
            self.listen_btn.configure(state=tk.NORMAL)
            self.speak_btn.configure(state=tk.NORMAL)
            self.confirm_btn.configure(state=tk.NORMAL)
            self.next_btn.configure(state=tk.NORMAL)
            return
        self._show_current_question()

    # ---------------- Prediction / Demo / Batch ----------------
    def _assemble_user_dataframe(self):
        # map responses to dataframe columns expected by your model
        inputs = {
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'stressLevel.value': self.responses.get('stress', 5),
            'mood.value': self.responses.get('mood', 5),
            'foodIntake.skippedBreakfast': self.responses.get('skipped_breakfast', 0),
            'foodIntake.skippedLunch': self.responses.get('skipped_lunch', 0),
            'foodIntake.skippedDinner': self.responses.get('skipped_dinner', 0),
            'activity_index': self.responses.get('activity', 50),
            'intensity': 0,
            'duration_m': 0
        }
        df = pd.DataFrame([inputs])
        return df

    def _on_predict(self):
        # run prediction using model if available
        if not self.model or not hasattr(self.model, 'predict_with_recommendations'):
            messagebox.showinfo("Model not available", "Prediction model not available. You can still test voice and UI.")
            self._log("Prediction aborted - model not available.")
            return

        user_df = self._assemble_user_dataframe()
        symptoms = self.responses.get('symptoms', "")
        triggers = self.responses.get('triggers', "")

        self._update_status("Analyzing...")

        # run model in thread to avoid blocking UI
        def worker():
            try:
                result = self.model.predict_with_recommendations(user_df, gender='female',
                                                                 user_symptoms=symptoms,
                                                                 user_triggers=triggers)
                # store history
                self.history.append({'timestamp': datetime.now(), 'result': result, 'inputs': self.responses.copy()})

                # announce using TTS (blocking) but invoked on a thread so UI stays responsive
                # show and speak result
                summary = f"Risk: {result['risk_label']} with probability {result['probability']} percent."
                print("[MODEL] result:", result)
                # use speak_blocking to ensure no STT interference
                self.voice.speak_blocking(summary, on_start=lambda: self._set_tts_indicator("speaking"),
                                          on_end=lambda: self._set_tts_indicator("idle"))

                # update GUI with result (on main thread)
                def _show():
                    self._log(f"Prediction: {result['risk_label']} ({result['probability']}%)")
                    messagebox.showinfo("Prediction Result", f"{result['risk_label']} ({result['probability']}%)\n\n{result.get('risk_message','')}")
                    self._update_status("Prediction complete.")
                self.after(10, _show)

            except Exception as e:
                self.after(10, lambda: messagebox.showerror("Prediction error", str(e)))
                self._log(f"Prediction error: {e}")
                self._update_status("Prediction failed.")

        threading.Thread(target=worker, daemon=True).start()

    def _on_demo(self):
        # populate demo responses and run predict (without waiting for user)
        self.responses = {
            'stress': 8, 'mood': 3, 'skipped_breakfast': 1,
            'skipped_lunch': 0, 'skipped_dinner': 0, 'activity': 25,
            'symptoms': 'light sensitivity, fatigue',
            'triggers': 'stress, missed meals'
        }
        self._log("Demo responses prefilled.")
        # Auto-run prediction (will notify if model missing)
        self._on_predict()

    def _on_batch_predict(self):
        # ask for CSV file path
        file_path = filedialog.askopenfilename(title="Select CSV for batch prediction", filetypes=[("CSV files","*.csv"),("All files","*.*")])
        if not file_path:
            return
        try:
            df = pd.read_csv(file_path)
            self._log(f"Loaded {len(df)} rows for batch prediction.")
            if not self.model or not hasattr(self.model, 'predict_with_recommendations'):
                messagebox.showinfo("Model missing", "Model not loaded. Cannot run predictions.")
                return

            results = []
            def worker():
                for i, row in df.iterrows():
                    user_df = pd.DataFrame([row])
                    try:
                        r = self.model.predict_with_recommendations(user_df, gender='female')
                        results.append(r)
                        if i < 3:
                            self._log(f"Batch [{i+1}] -> {r['risk_label']} ({r['probability']}%)")
                    except Exception as e:
                        self._log(f"Batch predict error row {i}: {e}")
                self._log("Batch prediction finished.")
                messagebox.showinfo("Batch Done", f"Processed {len(results)} records.")
            threading.Thread(target=worker, daemon=True).start()

        except Exception as e:
            messagebox.showerror("CSV load error", str(e))

    def _show_history(self):
        if not self.history:
            messagebox.showinfo("History", "No prediction history yet.")
            return
        text = ""
        for i, h in enumerate(self.history[::-1], 1):
            res = h['result']
            text += f"{i}. {h['timestamp'].strftime('%Y-%m-%d %H:%M')} - {res.get('risk_label')} ({res.get('probability')}%)\n"
        messagebox.showinfo("Prediction History", text)

    def _on_exit(self):
        self.voice.stop()
        self.destroy()

# --------------- Visualization helper (preserved) ----------------
def create_visualization(result, user_inputs):
    """Create and save a visualization image - adapted from your previous CLI code."""
    try:
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Migraine Risk Analysis Dashboard', fontsize=16, fontweight='bold')

        # 1. Risk Gauge
        risk_levels = ['Low', 'Moderate', 'High', 'Very High']
        risk_thresholds = [20, 50, 70, 100]
        colors = ['#2ecc71', '#f1c40f', '#e67e22', '#e74c3c']
        current_risk = result.get('probability', 0)
        risk_index = next((i for i, t in enumerate(risk_thresholds) if current_risk <= t), len(risk_thresholds)-1)
        ax1.barh(risk_levels, risk_thresholds, color=colors, alpha=0.3)
        ax1.barh(risk_levels[risk_index], risk_thresholds[risk_index], color=colors[risk_index], alpha=0.7)
        ax1.axvline(current_risk, color='black', linestyle='--', alpha=0.8)
        ax1.text(current_risk + 2, risk_index, f'{current_risk}%', va='center', fontweight='bold')
        ax1.set_xlabel('Probability (%)')
        ax1.set_title('Migraine Risk Level')

        # 2. Factor analysis (best-effort)
        factors = {
            'Stress': user_inputs.get('stress', 0),
            'Mood (inverted)': max(0, 10 - user_inputs.get('mood', 5)),
            'Meals Skipped': sum([user_inputs.get('skipped_breakfast', 0), user_inputs.get('skipped_lunch', 0), user_inputs.get('skipped_dinner', 0)]) * 3.33,
            'Low Activity': max(0, 50 - user_inputs.get('activity', 50)) / 5
        }
        names = list(factors.keys())
        vals = list(factors.values())
        bars = ax2.bar(names, vals, alpha=0.7)
        ax2.set_title('Risk Factor Analysis')
        ax2.tick_params(axis='x', rotation=30)
        for bar, v in zip(bars, vals):
            ax2.text(bar.get_x() + bar.get_width()/2., v + 0.1, f'{v:.1f}', ha='center')

        # 3. Trend (simulated)
        days = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun']
        trend = [random.randint(10, 90) for _ in range(6)] + [current_risk]
        ax3.plot(days, trend, marker='o')
        ax3.fill_between(days, trend, alpha=0.2)
        ax3.set_ylim(0, 100)
        ax3.set_title('Weekly Risk Trend')

        # 4. Recommendations (if present)
        recs = result.get('key_recommendations', [])
        if recs:
            rec_names = [r['action'][:30] + ("..." if len(r['action'])>30 else "") for r in recs]
            urg_map = {'high':3, 'medium':2, 'low':1}
            urg_vals = [urg_map.get(r.get('urgency','low'),1) for r in recs]
            ax4.barh(rec_names, urg_vals)
            ax4.set_title('Recommendations Priority')

        plt.tight_layout()
        fname = 'migraine_analysis.png'
        plt.savefig(fname, dpi=150, bbox_inches='tight')
        return fname
    except Exception as e:
        print("Visualization error:", e)
        return None

# ----------------- Main -----------------
def main():
    app = MigraineGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
