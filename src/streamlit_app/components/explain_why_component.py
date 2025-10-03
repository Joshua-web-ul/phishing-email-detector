import streamlit as st

def explain_why_component(prediction, explanation, email_text):
    with st.expander("Explain Why", expanded=True):
        if prediction == 1:
            st.markdown("### This email is likely a phishing attempt.")
        else:
            st.markdown("### This email is likely legitimate.")

        # Display the email text
        st.markdown("#### Email Content:")
        st.markdown(email_text)

        # Display the explanation
        st.markdown("### Explanation:")
        st.markdown(explanation)
