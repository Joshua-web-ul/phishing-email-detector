from streamlit import expander, markdown

def explain_why(prediction, shap_values, feature_names, email_text):
    with expander("Explain Why", expanded=True):
        if prediction == 1:
            markdown("### This email is likely a phishing attempt.")
        else:
            markdown("### This email is likely legitimate.")

        # Display the email text
        markdown("#### Email Content:")
        markdown(email_text)

        # Display SHAP values
        markdown("### Important Features:")
        for i in shap_values.argsort()[0][:10]:  # Show top 10 features
            markdown(f"- **{feature_names[i]}**: {shap_values[0][i]:.4f}")

        markdown("### Interpretation:")
        markdown("The above features contributed the most to the model's decision. A higher absolute SHAP value indicates a stronger influence on the prediction.")