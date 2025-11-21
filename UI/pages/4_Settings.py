# UI/pages/4_Settings.py
import streamlit as st

def render():
    """Render the settings page"""
    st.header("Settings")
    
    # User Profile
    st.subheader("User Profile")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("Full Name", placeholder="Enter your name")
        st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"])
    
    with col2:
        st.number_input("Age", min_value=18, max_value=100, value=30)
        st.date_input("Birth Date")
    
    # Known Triggers
    st.subheader("Known Triggers")
    
    trigger_cols = st.columns(3)
    triggers = [
        "Stress", "Sleep Changes", "Meal Timing", "Bright Lights",
        "Loud Noises", "Strong Smells", "Weather Changes", "Hormonal Changes"
    ]
    
    for i, trigger in enumerate(triggers):
        with trigger_cols[i % 3]:
            st.checkbox(trigger, value=(i % 2 == 0))
    
    # Notification Settings
    st.subheader("Notifications")
    st.checkbox("Enable risk alerts", value=True)
    st.checkbox("Daily reminder to check-in", value=True)
    st.checkbox("Weekly summary report", value=True)
    
    # Data Management
    st.subheader("Data Management")
    st.button("Export My Data")
    st.button("Clear History", type="secondary")
    
    # Save button
    if st.button("Save Settings", type="primary"):
        st.success("Settings saved successfully!")