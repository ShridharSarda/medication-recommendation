from flask import Flask, render_template, request, jsonify
import pandas as pd
import ast
import re
from fuzzywuzzy import fuzz
import nltk
from nltk.corpus import stopwords
import spacy
import os

# Load NLP models and stop words
try:
    nlp = spacy.load("en_core_web_md")  # Medium model for better accuracy
except OSError:
    from spacy.cli import download
    download("en_core_web_md")
    nlp = spacy.load("en_core_web_md")

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)

# File paths
DATA_DIR = "data"
FILES = {
    "sym_des": os.path.join(DATA_DIR, "symtoms_df.csv"),
    "precautions": os.path.join(DATA_DIR, "precautions_df.csv"),
    "medications": os.path.join(DATA_DIR, "medications.csv"),
    "diets": os.path.join(DATA_DIR, "diets.csv"),
    "workout": os.path.join(DATA_DIR, "workout_df.csv"),
    "description": os.path.join(DATA_DIR, "description.csv"),
}

# Load datasets
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    print(f"Error: File not found - {file_path}")
    return pd.DataFrame()  # Return an empty DataFrame if the file is missing

sym_des = load_data(FILES["sym_des"])
precautions = load_data(FILES["precautions"])
medications = load_data(FILES["medications"])
diets = load_data(FILES["diets"])
workout = load_data(FILES["workout"])
description = load_data(FILES["description"])

# List of known symptoms
known_symptoms = set()
if not sym_des.empty:
    for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']:
        known_symptoms.update(sym_des[col].dropna().str.lower().str.strip().unique())

# Preprocess text function
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())  # Lowercase, remove special chars
    text = " ".join([word for word in text.split() if word not in stop_words])  # Remove stop words
    return text

# Match symptoms using NLP
def match_symptom_with_nlp(user_input, known_symptoms):
    user_input_processed = preprocess_text(user_input)
    user_doc = nlp(user_input_processed)
    best_match = None
    highest_similarity = 0

    for symptom in known_symptoms:
        symptom_doc = nlp(symptom)
        similarity = user_doc.similarity(symptom_doc)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = symptom

    return best_match if highest_similarity > 0.7 else None  # Threshold for similarity

# Match symptoms using Fuzzy Matching
def match_symptom_with_fuzzy(user_input, known_symptoms):
    user_input_processed = preprocess_text(user_input)
    best_match, highest_score = None, 0

    for symptom in known_symptoms:
        score = fuzz.partial_ratio(user_input_processed, symptom)
        if score > highest_score:
            highest_score = score
            best_match = symptom

    return best_match if highest_score > 80 else None  # Threshold for fuzzy match

# Extract symptoms from user input
def extract_symptoms(user_input):
    user_input = preprocess_text(user_input)
    symptoms_found = []

    # Split input into phrases
    phrases = re.split(r',|\band\b', user_input)

    for phrase in phrases:
        matched_symptom = match_symptom_with_nlp(phrase.strip(), known_symptoms) or \
                          match_symptom_with_fuzzy(phrase.strip(), known_symptoms)
        if matched_symptom:
            symptoms_found.append(matched_symptom)

    return list(set(symptoms_found))  # Remove duplicates

# Match user symptoms with diseases
def match_symptoms(user_symptoms):
    disease_match_scores = {}

    for _, row in sym_des.iterrows():
        disease_symptoms = [
            str(row[col]).lower().strip()
            for col in ['Symptom_1', 'Symptom_2', 'Symptom_3', 'Symptom_4']
            if pd.notnull(row[col])
        ]
        match_count = sum(1 for symptom in user_symptoms if symptom in disease_symptoms)
        if match_count > 0:
            disease = row['Disease']
            disease_match_scores[disease] = match_count

    sorted_diseases = sorted(disease_match_scores.items(), key=lambda x: x[1], reverse=True)
    return [disease for disease, _ in sorted_diseases]

# Fetch precautions for a disease
def get_precautions(disease_name):
    result = precautions[precautions["Disease"].str.lower() == disease_name.lower()]
    if not result.empty:
        return result.iloc[0][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].dropna().tolist()
    return ["No precautions found for this disease."]

# Get detailed disease info
def get_disease_info(diseases):
    disease_info = []

    for disease in diseases:
        # Description
        desc_row = description[description['Disease'] == disease]
        description_text = desc_row['Description'].values[0] if not desc_row.empty else "Description not available"

        # Precautions
        pre = get_precautions(disease)

        # Medications
        med_row = medications[medications['Disease'] == disease]
        med_str = med_row['Medication'].values[0] if not med_row.empty else "Medication data not available"
        try:
            med = ast.literal_eval(med_str) if isinstance(med_str, str) else med_str
        except (ValueError, SyntaxError):
            med = [med_str]

        # Diet
        die_row = diets[diets['Disease'] == disease]
        die_str = die_row['Diet'].values[0] if not die_row.empty else "Diet data not available"
        try:
            die = ast.literal_eval(die_str) if isinstance(die_str, str) else die_str
        except (ValueError, SyntaxError):
            die = [die_str]

        # Workout
        workout_row = workout[workout['disease'] == disease]
        wrkout = workout_row['workout'].values[0] if not workout_row.empty else "Workout information not available"

        disease_info.append({
            'disease': disease,
            'description': description_text,
            'precautions': pre,
            'medications': med,
            'diet': die,
            'workout': wrkout
        })

    return disease_info

# Routes
@app.route("/chat", methods=["POST"])
def chat():
    user_input = request.json["message"].lower()

    if "hi" in user_input or "hello" in user_input:
        response = "Hi! Hope you're good. Tell me your symptoms to find a possible disease."
    else:
        user_symptoms = extract_symptoms(user_input)
        if user_symptoms:
            matched_diseases = match_symptoms(user_symptoms)
            if matched_diseases:
                # Create an unordered list for all matched diseases
                response_parts = "<p>I think you might have the following diseases:</p><ul>"
                for disease in matched_diseases:
                    response_parts += f"<li><a href='/disease/{disease}' class='disease-link'>{disease}</a></li>"
                response_parts += "</ul>"

                # Combine responses
                response = response_parts
            else:
                response = "Sorry, I couldn't match any disease with your symptoms. Try describing differently."
        else:
            response = "I couldn't recognize any symptoms. Please try again."

    return jsonify({"response": response})



@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    symptoms_input = request.form['symptoms']
    user_symptoms = extract_symptoms(symptoms_input)
    matched_diseases = match_symptoms(user_symptoms)

    if matched_diseases:
        disease_info = get_disease_info(matched_diseases)
    else:
        disease_info = []

    return render_template('results.html', matched_diseases=matched_diseases, disease_info=disease_info)

@app.route('/disease/<disease_name>')
def disease_details(disease_name):
    disease_info = get_disease_info([disease_name])
    return render_template('results.html', matched_diseases=[disease_name], disease_info=disease_info)

# Main execution block
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)