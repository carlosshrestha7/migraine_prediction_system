# UI/components/risk_display.py - COLOR-CODED RISK BANDS (Matches UI Flow Diagram)
import streamlit as st
from utils.config import APP_CONFIG

def render_risk_assessment(prediction_data=None):
    """Render color-coded risk assessment display"""
    
    if not prediction_data:
        # Show placeholder
        st.info("Complete a daily check-in to see your risk assessment")
        return
    
    risk_band = prediction_data['risk_band']
    risk_info = APP_CONFIG['RISK_BANDS'][risk_band]
    
    # Risk header with color coding
    st.subheader(f"{risk_info['color']} {prediction_data['risk_label']}")
    
    # Progress bar
    progress_value = prediction_data['probability'] / 100
    st.progress(progress_value)
    
    # Metrics in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Probability", 
            value=f"{prediction_data['probability']}%"
        )
    
    with col2:
        st.metric(
            label="Risk Level", 
            value=prediction_data['risk_label']
        )
    
    with col3:
        intensity = prediction_data.get('intensity_label', 'None')
        st.metric(
            label="Predicted Intensity", 
            value=intensity
        )
    
    # Risk message
    st.write(f"**Assessment:** {prediction_data['risk_message']}")

def render_risk_guide():
    """Render the risk guide legend"""
    st.subheader("Risk Guide")
    for band, info in APP_CONFIG['RISK_BANDS'].items():
        st.write(f"{info['color']} **{info['label']}** ({info['min']}-{info['max']}%)")