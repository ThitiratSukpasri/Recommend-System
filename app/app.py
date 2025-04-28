from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import os 

# Initialize Flask app
app = Flask(__name__)

# Load and prepare data
# Get the current working directory (project root directory)
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Define the relative path to the CSV file
csv_file_path = os.path.join(project_dir, 'data', 'symptom_data.csv')

# Load the CSV file
df = pd.read_csv(csv_file_path)

# Preprocess symptoms
df['symptom_list'] = df['search_term'].apply(lambda x: [s.strip() for s in str(x).split(',') if s.strip()])

# Build symptom vocabulary
all_symptoms = set()
for symptoms in df['symptom_list']:
    all_symptoms.update(symptoms)
all_symptoms = sorted(list(all_symptoms))

# Create patient-symptom matrix
patient_symptom_matrix = []
for symptoms in df['symptom_list']:
    row = [1 if symptom in symptoms else 0 for symptom in all_symptoms]
    patient_symptom_matrix.append(row)
patient_symptom_matrix = np.array(patient_symptom_matrix)

# Train KNN model
knn = NearestNeighbors(n_neighbors=30, metric='cosine')
knn.fit(patient_symptom_matrix)

# Define recommendation function
def recommend_symptoms_knn(input_symptoms, top_k=5):
    input_vector = np.array([1 if symptom in input_symptoms else 0 for symptom in all_symptoms]).reshape(1, -1)
    distances, indices = knn.kneighbors(input_vector)

    symptom_counter = {}
    for idx in indices[0]:
        patient_symptoms = df.iloc[idx]['symptom_list']
        for symptom in patient_symptoms:
            if symptom not in input_symptoms:
                symptom_counter[symptom] = symptom_counter.get(symptom, 0) + 1

    sorted_recommendations = sorted(symptom_counter.items(), key=lambda x: x[1], reverse=True)

    return [symptom for symptom, count in sorted_recommendations[:top_k]]

# API route


#@app.route('/test', methods=['GET'])
#def test():
    #return jsonify({'message': 'Server is working!'})

@app.route('/app', methods=['POST'])
def recommend():
    data = request.get_json()
    input_symptoms = data.get('symptoms', [])

    if not input_symptoms:
        return jsonify({'error': 'Please provide a list of symptoms'}), 400

    recommendations = recommend_symptoms_knn(input_symptoms)
    return jsonify({'recommendations': recommendations})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
