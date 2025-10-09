# TODO: Implement Pending Enhancements for Phishing Email Detector

## 1. Hyperparameter Tuning
- [x] Implement GridSearchCV for Logistic Regression model in baseline_training.py.
- [x] Implement Optuna hyperparameter tuning for BERT model in bert_training.py.
- [x] Update baseline_training.py to train LogisticRegressionModel with combined text and features, saving fitted components to expected paths.

## 2. Data Augmentation
- [ ] Add synonym replacement or back-translation augmentation in data preprocessing pipeline.

## 3. Model Quantization
- [ ] Convert fine-tuned BERT model to ONNX format.
- [ ] Apply quantization to ONNX model for faster inference.

## 4. Caching
- [ ] Implement LRU caching for prediction results in Streamlit app to speed up repeated queries.

## 5. Ensemble Model
- [ ] Create a voting ensemble combining Logistic Regression and BERT predictions.
- [ ] Update training and inference scripts to support ensemble.

## 6. Advanced Architectures
- [ ] Explore and implement DistilBERT, CNN, and LSTM models for phishing detection.

## 7. User Feedback Loop
- [ ] Add UI components to allow users to correct predictions.
- [ ] Store feedback data for active learning and model retraining.

## 8. Batch Processing
- [ ] Add functionality to upload and analyze multiple emails at once in Streamlit UI.

## 9. Real-time Monitoring
- [ ] Implement logging of prediction requests and model performance metrics.
- [ ] Create dashboards or reports for monitoring.

## 10. API Endpoint
- [ ] Develop REST API for phishing detection service.
- [ ] Document API endpoints and usage.

## Follow-up Steps
- [ ] Implement each feature incrementally with testing.
- [ ] Update documentation and user guides.
- [ ] Perform thorough testing after each major addition.
