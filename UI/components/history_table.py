# UI/components/history_table.py - PREDICTION HISTORY (Matches Database Schema)
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta

def render_history_table():
    """Render prediction history table"""
    
    # Mock history data (will connect to MIGRAINE_PREDICTION table later)
    history_data = [
        {
            'date': '2024-01-15',
            'probability': 15,
            'risk_level': 'Low',
            'intensity': 'None',
            'was_accurate': True,
            'factors': 'Low stress, good sleep'
        },
        {
            'date': '2024-01-14', 
            'probability': 45,
            'risk_level': 'Moderate',
            'intensity': 'Light',
            'was_accurate': True,
            'factors': 'High stress, missed meals'
        },
        {
            'date': '2024-01-13',
            'probability': 25, 
            'risk_level': 'Low',
            'intensity': 'None',
            'was_accurate': True,
            'factors': 'Normal day'
        }
    ]
    
    df = pd.DataFrame(history_data)
    
    # Display as interactive table
    st.dataframe(
        df,
        use_container_width=True,
        column_config={
            "date": "Date",
            "probability": st.column_config.NumberColumn(
                "Probability %",
                format="%d%%"
            ),
            "risk_level": "Risk Level",
            "intensity": "Intensity", 
            "was_accurate": "Accurate",
            "factors": "Contributing Factors"
        }
    )
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(history_data))
    with col2:
        accurate_predictions = sum(1 for item in history_data if item['was_accurate'])
        st.metric("Accuracy Rate", f"{(accurate_predictions/len(history_data))*100:.1f}%")
    with col3:
        avg_risk = sum(item['probability'] for item in history_data) / len(history_data)
        st.metric("Average Risk", f"{avg_risk:.1f}%")