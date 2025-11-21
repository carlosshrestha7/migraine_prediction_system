# UI/pages/3_History.py
import streamlit as st
from UI.components.history_table import render_history_table
from UI.components.risk_display import render_risk_guide

def render():
    """Render the history page"""
    st.header("Prediction History")
    st.markdown("Review your historical predictions and patterns")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        time_filter = st.selectbox(
            "Time Period",
            ["Last 7 days", "Last 30 days", "Last 90 days", "All time"]
        )
    with col2:
        risk_filter = st.selectbox(
            "Risk Level", 
            ["All", "Low", "Moderate", "High", "Very High"]
        )
    with col3:
        accuracy_filter = st.selectbox(
            "Accuracy",
            ["All", "Accurate", "Inaccurate"]
        )
    
    # History table
    render_history_table()
    
    # Risk guide for reference
    st.markdown("---")
    render_risk_guide()