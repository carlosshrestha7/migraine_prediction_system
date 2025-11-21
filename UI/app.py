# UI/app.py - UPDATED MAIN APP
import streamlit as st
from components.sidebar import render_sidebar, handle_page_navigation
from utils.config import APP_CONFIG

def main():
    # Page configuration
    st.set_page_config(
        page_title=APP_CONFIG['APP_TITLE'],
        page_icon=APP_CONFIG['APP_ICON'],
        layout=APP_CONFIG['LAYOUT'],
        initial_sidebar_state=APP_CONFIG['SIDEBAR_STATE']
    )
    
    # Initialize session state
    if 'current_page' not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Render sidebar
    render_sidebar()
    
    # Main title
    st.title(f"{APP_CONFIG['APP_ICON']} {APP_CONFIG['APP_TITLE']}")
    st.markdown(APP_CONFIG['APP_DESCRIPTION'])
    
    # Handle page navigation
    handle_page_navigation()

if __name__ == "__main__":
    main()git 