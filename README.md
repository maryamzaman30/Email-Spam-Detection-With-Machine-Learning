# Data Science Intern - Data Zenix Solutions

This project is a part of my **Data Science Internship** at **Data Zenix Solutions**, Karachi.

## Internship Details

- **Company:** Data Zenix Solutions, Karachi 🇵🇰
- **Internship Period:** September 2025

# Email Spam Detection System

## Objective
This project aims to develop a machine learning model that can accurately classify emails as either spam or not spam (ham). The system is implemented as a Streamlit web application, allowing users to input email content and receive an immediate prediction about its classification.

- Link to App Online: click the link in the About section on the left
- Dataset source - [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)

## Methodology / Approach

### Data Preprocessing
1. **Text Cleaning**: 
   - Converted text to lowercase
   - Removed special characters and numbers
   - Tokenized the text into individual words
   - Removed stop words using NLTK's English stopwords

### Model Training
1. **Feature Extraction**:
   - Used TF-IDF (Term Frequency-Inverse Document Frequency) for text vectorization
   - Transformed text data into numerical features

2. **Model Selection**:
   - Implemented a machine learning classifier (as seen in `spam_detection.ipynb`)
   - The model is saved as `spam_model.pkl` for deployment

> **Note:**  
> The `.pkl` files (`spam_model.pkl` and `vectorizer.pkl`) are not stored in this repository. To use the Streamlit app, first run `spam_detection.ipynb` to train the model and save the necessary files locally.

### Web Application
- Built with Streamlit for a user-friendly interface
- Processes user input in real-time
- Displays prediction results instantly

## Key Results and Observations

### Model Performance
- **Accuracy**: 0.9729
- **Precision**: 0.98 (macro avg), 0.97 (weighted avg)
- **Recall**: 0.90 (macro avg), 0.97 (weighted avg)
- **F1-Score**: 0.94 (macro avg), 0.97 (weighted avg)

### Key Findings
1. The model effectively differentiates between spam and legitimate emails
2. Most common spam indicators include:
   - Urgent action words
   - Suspicious links
   - Unusual sender addresses
   - Request for personal information

### Usage
1. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run `spam_detection.ipynb` file to train & save the model

4. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

5. Open your browser and navigate to the provided local URL

### Dependencies
- Python 3.x
- Streamlit
- scikit-learn
- NLTK
- joblib
- pandas (for data processing)
- numpy

### Files
- `app.py`: Main application file

- `spam_detection.ipynb`: Jupyter notebook with model development and analysis

- `spam_model.pkl`: Trained model ⚠️ *Not included in this repository due to size and versioning concerns.* You can generate it by running the notebook `spam_detection.ipynb`

- `vectorizer.pkl`: TF-IDF vectorizer ⚠️ *Not included in this repository due to size and versioning concerns.* You can generate it by running the notebook `spam_detection.ipynb`

- `spam.csv`: Dataset used for training