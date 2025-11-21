# UI/components/checkin_form.py - DAILY CHECK-IN FORM (Matches Comprehensive Trigger Categories)
import streamlit as st
from UI.utils.config import APP_CONFIG


def render_quick_checkin():
    """Render quick check-in form for dashboard"""
    with st.form("quick_checkin"):
        st.subheader("Quick Check-in")
        
        col1, col2 = st.columns(2)
        
        with col1:
            stress = st.slider("😰 Stress", 0, 10, 5, key="quick_stress")
            sleep = st.slider("😴 Sleep Hours", 0.0, 12.0, 7.0, 0.5, key="quick_sleep")
        
        with col2:
            mood = st.slider("😊 Mood", 0, 10, 7, key="quick_mood")
            activity = st.slider("💪 Activity", 0, 100, 50, key="quick_activity")
        
        submitted = st.form_submit_button("Quick Assessment")
        
        if submitted:
            return {
                'stress_level': stress,
                'sleep_hours': sleep,
                'mood': mood,
                'activity_level': activity
            }
    return None


def render_complete_checkin_form():
    """Render complete daily check-in form"""
    user_data = {}
    
    st.subheader("Profile Information")
    col1, col2 = st.columns(2)
    
    with col1:
        user_data['gender'] = st.selectbox(
            "Gender", 
            ["female", "male", "other"],
            key="checkin_gender"
        )
    
    with col2:
        user_data['age'] = st.number_input(
            "Age", 
            min_value=18, max_value=100, value=30,
            key="checkin_age"
        )
    
    st.subheader("Daily Factors")
    
    # Stress and Mood
    col1, col2 = st.columns(2)
    with col1:
        user_data['stress_level'] = st.slider(
            "Stress Level (0-10)", 
            0, 10, 5,
            help="0 = No stress, 10 = Extreme stress",
            key="checkin_stress"
        )
    
    with col2:
        user_data['mood'] = st.slider(
            "Mood (0-10)", 
            0, 10, 7,
            help="0 = Very poor, 10 = Excellent",
            key="checkin_mood"
        )
    
    # Sleep and Activity
    col1, col2 = st.columns(2)
    with col1:
        user_data['sleep_hours'] = st.slider(
            "Sleep Hours", 
            0.0, 12.0, 7.0, 0.5,
            key="checkin_sleep"
        )
    
    with col2:
        user_data['activity_level'] = st.slider(
            "Activity Level", 
            0, 100, 50,
            key="checkin_activity"
        )
    
    st.subheader("Nutrition")
    col1, col2, col3 = st.columns(3)
    with col1:
        user_data['had_breakfast'] = st.checkbox("Had Breakfast", key="breakfast")
    with col2:
        user_data['had_lunch'] = st.checkbox("Had Lunch", key="lunch")
    with col3:
        user_data['had_dinner'] = st.checkbox("Had Dinner", key="dinner")
    
    st.subheader("Symptoms & Triggers")
    
    col1, col2 = st.columns(2)
    with col1:
        user_data['symptoms'] = st.multiselect(
            "Current Symptoms",
            APP_CONFIG['SYMPTOMS'],
            key="checkin_symptoms"
        )
    
    with col2:
        user_data['triggers'] = st.multiselect(
            "Triggers Encountered",
            APP_CONFIG['TRIGGERS'],
            key="checkin_triggers"
        )
    
    # Additional notes
    user_data['notes'] = st.text_area(
        "Additional Notes",
        placeholder="Any other observations...",
        key="checkin_notes"
    )
    
    return user_datap