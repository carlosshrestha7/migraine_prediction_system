# UI/pages/2_Daily_Checkin.py
import streamlit as st
from UI.components.checkin_form import render_complete_checkin_form
from UI.services.prediction_service import get_prediction

def render():
    """Render the daily check-in page"""
    st.header("Daily Check-in")
    st.markdown("Complete your daily health assessment")
    
    # Render the complete check-in form
    user_data = render_complete_checkin_form()
    
    # Prediction button
    if st.button("Get Risk Prediction", type="primary", use_container_width=True):
        if validate_form(user_data):
            with st.spinner("Analyzing your data..."):
                # Get prediction
                prediction_result = get_prediction(user_data)
                display_prediction_results(prediction_result)
        else:
            st.error("Please fill in all required fields")

def validate_form(user_data):
    """Validate the check-in form data"""
    required_fields = ['stress_level', 'sleep_hours', 'mood']
    return all(user_data.get(field) for field in required_fields)

def display_prediction_results(prediction):
    """Display prediction results"""
    st.success("Prediction complete!")
    
    # Results in columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Risk Probability", f"{prediction['probability']}%")
    
    with col2:
        st.metric("Risk Level", prediction['risk_label'])
    
    with col3:
        st.metric("Intensity", prediction.get('intensity_label', 'None'))
    
    # Progress bar
    st.progress(prediction['probability'] / 100)
    
    # Recommendations
    st.subheader("Recommended Actions")
    for recommendation in prediction.get('recommendations', []):
        priority_icon = "🔴" if recommendation['priority'] == 'high' else "🟡" if recommendation['priority'] == 'medium' else "🟢"
        st.write(f"{priority_icon} **{recommendation['category']}**: {recommendation['action']}")