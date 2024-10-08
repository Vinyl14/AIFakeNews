from flask import Flask, render_template, request, redirect, url_for

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from imblearn.over_sampling import RandomOverSampler

app = Flask(__name__)

# Load the dataset
data = pd.read_csv("DATASET_BERITA_ASLI_PALSU - Sheet1.csv")

# Create binary labels
data['PALSU'] = (data['LABEL'] == "ASLI").astype(int)

# Drop unnecessary columns
data = data.drop(["LABEL"], axis=1)

# Split the data into training and testing sets
X, y = data['TEKS'], data['PALSU']

# Vectorize the text data
vectorizer = TfidfVectorizer(stop_words="english", max_df=0.7)
X_vectorized = vectorizer.fit_transform(X)

# Address class imbalance using oversampling
oversampler = RandomOverSampler(random_state=42)
X_balanced, y_balanced = oversampler.fit_resample(X_vectorized, y)

# Train a LinearSVC classifier
clf = LinearSVC(C=1.0)  # You can adjust the C parameter
clf.fit(X_balanced, y_balanced)

# Save the model and vectorizer
joblib.dump(clf, 'your_model_filename.pkl')
joblib.dump(vectorizer, 'your_vectorizer_filename.pkl')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        user_input_text = request.form['text']
        user_input_vectorized = vectorizer.transform([user_input_text])
        prediction = clf.predict(user_input_vectorized)

        result = "ASLI" if prediction[0] == 0 else "PALSU"
        return redirect(url_for('result', prediction=result, text=user_input_text))

@app.route('/result')
def result():
    prediction = request.args.get('prediction', '')
    text = request.args.get('text', '')
    return render_template('result.html', prediction=prediction, text=text)

if __name__ == '__main__':
    app.run(debug=True)
