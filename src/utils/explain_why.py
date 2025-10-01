from src.utils.shap_explainer import ShapExplainer

def explain_why(email_text):
    """
    Explain why the email was classified as phishing or legitimate using SHAP.
    """
    try:
        explainer = ShapExplainer('models/baseline_model.pkl', 'models/tfidf_vectorizer.pkl')
        explanation = explainer.explain(email_text)

        # Format the explanation
        explanation_text = "Explanation for the classification:\n\n"
        explanation_text += "Words contributing to phishing classification (positive SHAP values):\n"
        for word, val in explanation['top_positive_words']:
            explanation_text += f"- {word}: +{val:.3f}\n"

        explanation_text += "\nWords contributing to legitimate classification (negative SHAP values):\n"
        for word, val in explanation['top_negative_words']:
            explanation_text += f"- {word}: {val:.3f}\n"

        return explanation_text
    except Exception as e:
        return f"Error in explanation: {str(e)}"
