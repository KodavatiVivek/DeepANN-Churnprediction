# DeepANN-Churnprediction

## 📋 Overview
This project implements an Artificial Neural Network (ANN) to predict customer churn. Using deep learning techniques, the model identifies customers at risk of leaving a service or platform, enabling proactive retention strategies. This is a practical application of neural networks in business analytics and customer relationship management.

## 🎯 Use Cases
- **Customer Churn Prediction**: Identifying customers likely to cancel subscriptions or leave
- **Retention Strategy**: Developing targeted retention campaigns for at-risk customers
- **Business Analytics**: Understanding factors contributing to customer attrition
- **Risk Assessment**: Quantifying churn risk for different customer segments
- **Resource Allocation**: Prioritizing retention efforts based on churn probability
- **Revenue Protection**: Minimizing revenue loss from customer attrition
- **Customer Lifetime Value Prediction**: Estimating long-term customer value
- **Model Interpretability**: Understanding which features drive churn decisions

## 🛠️ Tech Stack

### Primary Language & Framework
- **Python** (3.6%) - Python scripting
- **Jupyter Notebook** (96.4%) - Interactive data science and model development

### Deep Learning & ML Libraries
- **TensorFlow/Keras** - Deep learning framework for ANN models
- **scikit-learn** - Machine learning algorithms and utilities
- **NumPy** - Numerical computing and array operations
- **Pandas** - Data manipulation and analysis
- **SciPy** - Scientific computing tools

### Data Processing & Visualization
- **Matplotlib** - Plotting and visualization
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations
- **Plotly Express** - High-level visualization API

### Data Preprocessing
- **StandardScaler/MinMaxScaler** - Feature normalization
- **LabelEncoder** - Categorical encoding
- **OneHotEncoder** - Categorical feature transformation

## 📁 Project Structure
```
DeepANN-Churnprediction/
├── notebooks/
│   ├── 01_data_exploration.ipynb           # Dataset overview and analysis
│   ├── 02_data_preprocessing.ipynb         # Data cleaning and preparation
│   ├── 03_eda.ipynb                        # Exploratory data analysis
│   ├── 04_feature_engineering.ipynb        # Feature creation and selection
│   ├── 05_ann_model_development.ipynb      # ANN architecture design
│   ├── 06_model_training.ipynb             # Training and hyperparameter tuning
│   └── 07_evaluation_results.ipynb         # Model evaluation and results
├── data/
│   ├── raw/                                # Original customer dataset
│   └── processed/                          # Cleaned and prepared data
├── models/
│   ├── saved_models/                       # Trained ANN models
│   └── model_checkpoints/                  # Training checkpoints
├── src/
│   ├── preprocessing.py                    # Data preprocessing functions
│   ├── model.py                            # ANN model architecture
│   ├── utils.py                            # Utility functions
│   └── evaluate.py                         # Evaluation metrics
├── results/
│   ├── plots/                              # Visualization outputs
│   ├── metrics/                            # Performance metrics
│   └── predictions/                        # Prediction results
└── requirements.txt
```

## 🚀 Getting Started

### Prerequisites
- Python 3.7 or higher
- Jupyter Notebook
- 4GB RAM minimum (8GB recommended)
- GPU support recommended for faster training

### Installation
```bash
# Clone the repository
git clone https://github.com/KodavatiVivek/DeepANN-Churnprediction.git
cd DeepANN-Churnprediction

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Running the Project
```bash
# Start Jupyter Notebook
jupyter notebook

# Open and run notebooks sequentially:
# 1. 01_data_exploration.ipynb
# 2. 02_data_preprocessing.ipynb
# 3. 03_eda.ipynb
# 4. 04_feature_engineering.ipynb
# 5. 05_ann_model_development.ipynb
# 6. 06_model_training.ipynb
# 7. 07_evaluation_results.ipynb
```

## 📚 Key Features

### ANN Architecture
- **Input Layer**: Accepts normalized customer features
- **Hidden Layers**: Multiple dense layers with ReLU/Sigmoid activation
- **Dropout Layers**: Prevents overfitting with 20-30% dropout rate
- **Batch Normalization**: Stabilizes training
- **Output Layer**: Sigmoid activation for binary classification

### Model Capabilities
- ✅ Binary classification (Churn vs. No Churn)
- ✅ Probability predictions
- ✅ Feature importance analysis
- ✅ Cross-validation support
- ✅ Hyperparameter tuning
- ✅ Model checkpointing
- ✅ Early stopping
- ✅ Comprehensive evaluation metrics

### Typical ANN Configuration
```python
{
    'input_features': 11,          # Number of input features
    'hidden_layers': [64, 32, 16], # Neurons in each hidden layer
    'activation': 'relu',           # Activation function
    'output_activation': 'sigmoid', # Output activation for binary classification
    'dropout_rate': 0.3,            # Dropout rate
    'batch_size': 32,               # Batch size
    'epochs': 100,                  # Training epochs
    'learning_rate': 0.001          # Learning rate
}
```

## 📊 Model Performance

### Expected Metrics
- **Accuracy**: 80-85% typical range
- **Precision**: Measures false positives
- **Recall**: Measures false negatives
- **F1-Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: Classification breakdown

### Evaluation Example
```
Classification Report:
              precision    recall  f1-score   support
        No Churn       0.82      0.89      0.85      1000
           Churn       0.76      0.63      0.69       400
        accuracy                           0.81      1400
       macro avg       0.79      0.76      0.77      1400
    weighted avg       0.80      0.81      0.80      1400
```

## 🔧 Configuration & Customization

### Feature Selection
Important features for churn prediction:
- Customer tenure
- Monthly charges
- Contract type
- Payment method
- Internet service type
- Customer service interactions

### Hyperparameter Tuning
```python
# Grid search for optimal parameters
param_grid = {
    'hidden_layers': [[64, 32], [128, 64, 32], [256, 128]],
    'learning_rate': [0.001, 0.01, 0.1],
    'dropout_rate': [0.2, 0.3, 0.4],
    'batch_size': [16, 32, 64]
}
```

## 📈 Results & Interpretation
- Training and validation curves
- Accuracy and loss evolution
- Confusion matrix visualization
- ROC curve and AUC score
- Feature importance ranking
- Churn probability distribution
- Business impact analysis

## 📖 Documentation
- [TensorFlow/Keras Guide](https://www.tensorflow.org/guide)
- [Keras Sequential Model](https://keras.io/guides/sequential_model/)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [ANN Best Practices](https://www.deeplearningbook.org/)

## 🧪 Testing & Validation
```bash
# Evaluate model on test set
python src/evaluate.py --model_path models/saved_models/best_model.h5 --data_path data/processed/test_data.csv

# Make predictions on new data
python src/predict.py --model_path models/saved_models/best_model.h5 --input_data customer_data.csv
```

## 💡 Business Applications
- Identify high-risk customers for targeted retention campaigns
- Optimize marketing spend on retention efforts
- Forecast revenue impact of churn
- Segment customers by churn probability
- Measure effectiveness of retention strategies
- Calculate customer lifetime value

## 🤝 Contributing
Contributions are welcome! Potential improvements:
- Additional feature engineering techniques
- Advanced ensemble methods
- Real-time prediction API
- Model serving and deployment
- A/B testing framework

## 📝 License
This project is open source and available under the MIT License.

## 👨‍💼 Author
**Kodavati Vivek** - Full-Stack Developer & Machine Learning Enthusiast

## 📧 Contact
- GitHub: [@KodavatiVivek](https://github.com/KodavatiVivek)

## 🌟 Acknowledgments
- TensorFlow and Keras communities
- Scikit-learn contributors
- Customer analytics research community

---

**Last Updated**: May 2026
**Repository**: [DeepANN-Churnprediction](https://github.com/KodavatiVivek/DeepANN-Churnprediction)
