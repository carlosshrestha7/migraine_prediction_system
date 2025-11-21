# UI/pages/1_Dashboard.py
import streamlit as st
from UI.components.risk_display import render_risk_assessment
from UI.components.checkin_form import render_quick_checkin

def render():
    """Render the main dashboard"""
    st.header("Dashboard Overview")
    
    # Main content area with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Current Risk Assessment
        st.subheader("Current Risk Assessment")
        render_risk_assessment()
        
        # Recent Activity
        st.subheader("Recent Activity")
        display_recent_activity()
    
    with col2:
        # Quick Check-in
        st.subheader("Quick Check-in")
        render_quick_checkin()
        
        # Triggers Overview
        st.subheader("Top Triggers")
        display_top_triggers()

def display_recent_activity():
    """Display recent user activity"""
    recent_data = [
        {"date": "2024-01-15", "risk": "Low", "symptoms": "None"},
        {"date": "2024-01-14", "risk": "Moderate", "symptoms": "Headache"},
        {"date": "2024-01-13", "risk": "Low", "symptoms": "None"},
    ]
    
    for activity in recent_data:
        with st.container():
            col1, col2, col3 = st.columns([2, 1, 2])
            with col1:
                st.write(f"**{activity['date']}**")
            with col2:
                st.write(activity['risk'])
            with col3:
                st.write(activity['symptoms'])
            st.markdown("---")

def display_top_triggers():
    """Display user's top triggers"""
    triggers = [
        {"trigger": "Stress", "frequency": "80%", "impact": "High"},
        {"trigger": "Sleep Changes", "frequency": "60%", "impact": "Medium"},
        {"trigger": "Meal Timing", "frequency": "40%", "impact": "Medium"},
    ]
    
    for trigger in triggers:
        st.metric(
            label=trigger['trigger'],
            value=trigger['frequency'],
            delta=trigger['impact']
        )