import streamlit as st

def display_footer():
    """
    Displays a consistent footer across all pages of the application
    """
    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f8f9fa;
        color: #6c757d;
        text-align: center;
        padding: 10px 0;
        border-top: 1px solid #dee2e6;
        font-size: 0.9em;
    }
    .footer a {
        color: #1E88E5;
        text-decoration: none;
        margin: 0 10px;
    }
    .footer a:hover {
        text-decoration: underline;
    }
    </style>
    <div class="footer">
        <div>© 2025 Phishing Email Detector | 
            <a href="#" target="_blank">Privacy Policy</a> | 
            <a href="#" target="_blank">Terms of Service</a> | 
            <a href="#" target="_blank">Contact Us</a>
        </div>
        <div style="font-size: 0.8em; margin-top: 5px;">
            Version 2.0.0 | Last updated: October 2025
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Add some padding to the bottom of the page to prevent content from being hidden behind the footer
    st.markdown("""
    <style>
    .main > div {
        padding-bottom: 80px !important;
    }
    </style>
    """, unsafe_allow_html=True)
