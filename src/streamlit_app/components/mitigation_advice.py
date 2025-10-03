import streamlit as st

def display_mitigation_advice(phishing_result):
    with st.expander("Mitigation Advice", expanded=True):
        if phishing_result:
            st.markdown("""
            **Mitigation Advice:**
            1. **Do not click on any links** or download attachments from the email.
            2. **Verify the sender's email address** by checking for any discrepancies.
            3. **Report the phishing attempt** to your email provider or IT department.
            4. **Delete the email** from your inbox to prevent accidental clicks in the future.
            5. **Educate yourself** about common phishing tactics to recognize future threats.
            """)
        else:
            st.markdown("""
            **No immediate action required.** This email appears to be legitimate. However, always remain vigilant and follow best practices for email security.
            """)
