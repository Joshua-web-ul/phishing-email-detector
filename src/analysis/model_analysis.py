import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix, 
    classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import eli5
from eli5.sklearn import PermutationImportance
import shap

class ModelAnalyzer:
    def __init__(self, model_path, vectorizer_path, test_data_path):
        """
        Initialize the ModelAnalyzer with paths to the model, vectorizer, and test data.
        
        Args:
            model_path (str): Path to the trained model file
            vectorizer_path (str): Path to the TF-IDF vectorizer
            test_data_path (str): Path to the test dataset
        """
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.test_data = pd.read_csv(test_data_path)
        self.X_test = None
        self.y_test = None
        self.y_pred = None
        self.y_pred_proba = None
        
    def prepare_data(self):
        """Prepare the test data for evaluation"""
        # Assuming the test data has 'text' and 'label' columns
        self.X_test = self.vectorizer.transform(self.test_data['text'])
        self.y_test = self.test_data['label']
        
    def evaluate_model(self):
        """Evaluate the model and return performance metrics"""
        if self.X_test is None:
            self.prepare_data()
            
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        self.y_pred_proba = self.model.predict_proba(self.X_test)[:, 1]
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, self.y_pred),
            'precision': precision_score(self.y_test, self.y_pred),
            'recall': recall_score(self.y_test, self.y_pred),
            'f1': f1_score(self.y_test, self.y_pred),
            'roc_auc': roc_auc_score(self.y_test, self.y_pred_proba),
            'avg_precision': average_precision_score(self.y_test, self.y_pred_proba)
        }
        
        # Classification report
        report = classification_report(self.y_test, self.y_pred, target_names=['Legitimate', 'Phishing'])
        
        return metrics, report
    
    def plot_confusion_matrix(self, save_path=None):
        """Generate and optionally save a confusion matrix"""
        if self.y_pred is None:
            self.evaluate_model()
            
        cm = confusion_matrix(self.y_test, self.y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_roc_curve(self, save_path=None):
        """Generate and optionally save an ROC curve"""
        if self.y_pred_proba is None:
            self.evaluate_model()
            
        fpr, tpr, _ = roc_curve(self.y_test, self.y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def plot_precision_recall_curve(self, save_path=None):
        """Generate and optionally save a precision-recall curve"""
        if self.y_pred_proba is None:
            self.evaluate_model()
            
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_pred_proba)
        avg_precision = average_precision_score(self.y_test, self.y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.step(recall, precision, where='post')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title(f'Precision-Recall curve: AP={avg_precision:0.2f}')
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
    
    def analyze_feature_importance(self, top_n=20):
        """Analyze and return the top N most important features"""
        if not hasattr(self.model, 'feature_importances_'):
            # For models without feature_importances_, use permutation importance
            perm_importance = PermutationImportance(self.model, random_state=42).fit(self.X_test, self.y_test)
            feature_importance = pd.DataFrame({
                'feature': self.vectorizer.get_feature_names_out(),
                'importance': perm_importance.importances_mean
            })
        else:
            # For models with feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.vectorizer.get_feature_names_out(),
                'importance': self.model.feature_importances_
            })
        
        # Sort by importance and get top N
        top_features = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title('Top {} Most Important Features'.format(top_n))
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/feature_importance.png')
        plt.close()
        
        return top_features
    
    def generate_shap_values(self, sample_size=100):
        """Generate SHAP values for model interpretation"""
        # Sample data for faster computation
        if sample_size < len(self.X_test):
            sample_idx = np.random.choice(self.X_test.shape[0], sample_size, replace=False)
            X_sample = self.X_test[sample_idx]
        else:
            X_sample = self.X_test
        
        # Create SHAP explainer
        explainer = shap.LinearExplainer(self.model, X_sample, feature_dependence="independent")
        shap_values = explainer.shap_values(X_sample)
        
        # Plot summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=self.vectorizer.get_feature_names_out(), show=False)
        plt.title('SHAP Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('reports', exist_ok=True)
        plt.savefig('reports/shap_summary.png')
        plt.close()
        
        return shap_values
    
    def generate_report(self, output_dir='reports'):
        """Generate a comprehensive model evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate all plots
        self.plot_confusion_matrix(os.path.join(output_dir, 'confusion_matrix.png'))
        self.plot_roc_curve(os.path.join(output_dir, 'roc_curve.png'))
        self.plot_precision_recall_curve(os.path.join(output_dir, 'precision_recall_curve.png'))
        
        # Get metrics and feature importance
        metrics, report = self.evaluate_model()
        feature_importance = self.analyze_feature_importance()
        
        # Generate HTML report
        html_content = f"""
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2, h3 {{ color: #2c3e50; }}
                .metrics {{ 
                    display: grid; 
                    grid-template-columns: repeat(3, 1fr); 
                    gap: 15px; 
                    margin: 20px 0; 
                }}
                .metric-card {{ 
                    background: #f8f9fa; 
                    padding: 15px; 
                    border-radius: 5px; 
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    text-align: center;
                }}
                .metric-value {{ 
                    font-size: 24px; 
                    font-weight: bold; 
                    color: #3498db;
                    margin: 10px 0;
                }}
                .images {{ 
                    display: flex; 
                    flex-wrap: wrap; 
                    gap: 20px; 
                    margin: 20px 0; 
                }}
                .image-container {{ 
                    flex: 1; 
                    min-width: 300px; 
                    text-align: center;
                }}
                .image-container img {{ 
                    max-width: 100%; 
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}
                table {{ 
                    width: 100%; 
                    border-collapse: collapse; 
                    margin: 20px 0; 
                }}
                th, td {{ 
                    padding: 12px; 
                    text-align: left; 
                    border-bottom: 1px solid #ddd;
                }}
                th {{ 
                    background-color: #f2f2f2; 
                    font-weight: bold;
                }}
                tr:hover {{ background-color: #f5f5f5; }}
            </style>
        </head>
        <body>
            <h1>Model Evaluation Report</h1>
            <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <h2>Performance Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <div>Accuracy</div>
                    <div class="metric-value">{metrics['accuracy']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div>Precision</div>
                    <div class="metric-value">{metrics['precision']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div>Recall</div>
                    <div class="metric-value">{metrics['recall']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div>F1 Score</div>
                    <div class="metric-value">{metrics['f1']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div>ROC AUC</div>
                    <div class="metric-value">{metrics['roc_auc']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div>Avg Precision</div>
                    <div class="metric-value">{metrics['avg_precision']:.4f}</div>
                </div>
            </div>
            
            <h2>Classification Report</h2>
            <pre>{report}</pre>
            
            <h2>Visualizations</h2>
            <div class="images">
                <div class="image-container">
                    <h3>Confusion Matrix</h3>
                    <img src="confusion_matrix.png" alt="Confusion Matrix">
                </div>
                <div class="image-container">
                    <h3>ROC Curve</h3>
                    <img src="roc_curve.png" alt="ROC Curve">
                </div>
                <div class="image-container">
                    <h3>Precision-Recall Curve</h3>
                    <img src="precision_recall_curve.png" alt="Precision-Recall Curve">
                </div>
                <div class="image-container">
                    <h3>Feature Importance</h3>
                    <img src="feature_importance.png" alt="Feature Importance">
                </div>
            </div>
            
            <h2>Top 20 Most Important Features</h2>
            {feature_importance.to_html(index=False, classes='feature-table')}
            
            <h2>Model Information</h2>
            <table>
                <tr>
                    <th>Model Type</th>
                    <td>{type(self.model).__name__}</td>
                </tr>
                <tr>
                    <th>Vectorizer</th>
                    <td>{type(self.vectorizer).__name__}</td>
                </tr>
                <tr>
                    <th>Number of Features</th>
                    <td>{self.X_test.shape[1]}</td>
                </tr>
                <tr>
                    <th>Test Samples</th>
                    <td>{len(self.y_test)}</td>
                </tr>
            </table>
        </body>
        </html>
        """
        
        # Save HTML report
        with open(os.path.join(output_dir, 'model_evaluation_report.html'), 'w') as f:
            f.write(html_content)
            
        print(f"Report generated successfully at {os.path.join(output_dir, 'model_evaluation_report.html')}")

# Example usage
if __name__ == "__main__":
    # Initialize the analyzer with paths to your model, vectorizer, and test data
    analyzer = ModelAnalyzer(
        model_path='models/baseline_model.pkl',
        vectorizer_path='models/tfidf_vectorizer.pkl',
        test_data_path='data/test_data.csv'  # Update this path to your test data
    )
    
    # Generate and save the report
    analyzer.generate_report()
    
    # You can also call individual methods as needed:
    # metrics, report = analyzer.evaluate_model()
    # analyzer.plot_confusion_matrix()
    # feature_importance = analyzer.analyze_feature_importance()
    # shap_values = analyzer.generate_shap_values()
