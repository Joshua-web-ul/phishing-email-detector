import pytest
from src.utils.shap_explainer import ShapExplainer
from src.utils.explain_why import explain_why

def test_shap_explainer():
    # Test ShapExplainer initialization and explain method
    explainer = ShapExplainer('models/baseline_model.pkl', 'models/tfidf_vectorizer.pkl')
    sample_text = "This is a test email."
    explanation = explainer.explain(sample_text)
    
    # Check if explanation is returned
    assert explanation is not None, "Explanation should not be None"
    assert 'top_positive_words' in explanation, "Explanation should contain top positive words"
    assert 'top_negative_words' in explanation, "Explanation should contain top negative words"
    assert isinstance(explanation['top_positive_words'], list), "Top positive words should be a list"
    assert isinstance(explanation['top_negative_words'], list), "Top negative words should be a list"

def test_explain_why():
    # Test the explain_why function
    sample_text = "Congratulations! You've won a prize. Click here."
    explanation = explain_why(sample_text)
    
    # Check if explanation is returned
    assert explanation is not None, "Explanation should not be None"
    assert isinstance(explanation, str), "Explanation should be a string"
    assert "Explanation for the classification:" in explanation, "Explanation should contain the header"
