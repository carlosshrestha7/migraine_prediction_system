# UI/components/sidebar.py - NAVIGATION (Matches your Use Case Diagram)
import streamlit as st
from UI.utils.config import APP_CONFIG

def render_sidebar():
    """Render the main navigation sidebar"""
    with st.sidebar:
        st.header(f"{APP_CONFIG['APP_ICON']} Navigation")
        
        # Page selection (from Use Case Diagram)
        page = st.radio(
            "Go to:",
            ["📊 Dashboard", "📝 Daily Check-in", "📈 History", "⚙️ Settings"],
            index=0
        )
        
        # Store current page in session state
        st.session_state.current_page = page
        
        # Quick stats (from Database Schema)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Quick Stats")
        st.sidebar.metric("Check-ins This Week", "5")
        st.sidebar.metric("Current Risk Level", "Low")
        
        # Risk guide (from UI Flow Diagram)
        st.sidebar.markdown("---")
        st.sidebar.subheader("Risk Guide")
        for band, info in APP_CONFIG['RISK_BANDS'].items():
            st.sidebar.markdown(f"{info['color']} **{info['label']}** ({info['min']}-{info['max']}%)")
        
        # Footer
        st.sidebar.markdown("---")
        st.sidebar.caption("Migraine Prediction System v1.0")

# Update main app to handle page navigation
def handle_page_navigation():
    """Handle page navigation based on sidebar selection"""
    current_page = st.session_state.get('current_page', '📊 Dashboard')
    
    if current_page == "📊 Dashboard":
        from pages import dashboard
        dashboard.render()
    elif current_page == "📝 Daily Check-in":
        from pages import daily_checkin
        daily_checkin.render()
    elif current_page == "📈 History":
        from pages import history
        history.render()
    elif current_page == "⚙️ Settings":
        from pages import settings
        settings.render()