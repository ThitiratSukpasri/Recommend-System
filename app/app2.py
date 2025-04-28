from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import pandas as pd
import os

# Initialize Flask app
app = Flask(__name__)
project_dir = os.path.abspath(os.path.join(os.getcwd(), '..'))

# Define the relative path to the CSV file
csv_file_path = os.path.join(project_dir, 'data', 'symptom_data.csv')

def get_symptom_recommendations_tfidf(df, input_symptoms, top_k=5):
    """
    Recommends symptoms similar to the input symptoms using TF-IDF and KNN.
    """
    try:
        # Preprocess symptoms
        if 'symptom_list' in df.columns:
            symptom_column = 'symptom_list'
        elif 'search_term' in df.columns:
            symptom_column = 'search_term'
        else:
            return []  # Return empty list if neither column exists

        # Ensure symptoms are not empty
        # Ensure symptoms are not empty
        df['symptom_string'] = df[symptom_column].apply(lambda x: ' '.join(x) if isinstance(x, list) else (x if isinstance(x, str) else 'default_symptom'))

        # Remove rows with default or missing symptoms
        df = df[df['symptom_string'] != 'default_symptom']
        df = df.dropna(subset=['symptom_string'])
        df['symptom_string'] = df['symptom_string'].str.replace(',', ' ').str.replace('  ', ' ').str.strip()

        print("Data after preprocessing:", df['symptom_string'].head())

        # Initialize the TF-IDF Vectorizer
        tfidf = TfidfVectorizer(stop_words=None, max_features=5000)

        # Fit and transform the symptoms into a TF-IDF matrix
        tfidf_matrix = tfidf.fit_transform(df['symptom_string'])

        # Train KNN model on the TF-IDF matrix
        knn = NearestNeighbors(n_neighbors=top_k, metric='cosine')
        knn.fit(tfidf_matrix)

        # Create input vector for the symptoms
        input_string = ' '.join(input_symptoms)
        input_vector = tfidf.transform([input_string])

        # Find nearest neighbors
        distances, indices = knn.kneighbors(input_vector, n_neighbors=top_k)

        # Count recommended symptoms
        symptom_counter = {}
        for idx in indices[0]:
            patient_symptoms = df.iloc[idx][symptom_column]  # Use the determined column
            for symptom in patient_symptoms:
                if symptom not in input_symptoms:
                    symptom_counter[symptom] = symptom_counter.get(symptom, 0) + 1

        sorted_recommendations = sorted(symptom_counter.items(), key=lambda x: x[1], reverse=True)
        return [symptom for symptom, count in sorted_recommendations[:top_k]]

    except Exception as e:
        print(f"An error occurred: {e}")
        return []

@app.route('/recommend', methods=['POST'])
def recommend():
    """
    API endpoint to recommend symptoms based on input symptoms using TF-IDF.
    """
    # Get the input data
    data = request.get_json()

    # Validate input
    input_symptoms = data.get('symptoms', [])
    if not input_symptoms:
        return jsonify({'error': 'Please provide a list of symptoms'}), 400

    # Load the data
    try:
        df = pd.read_csv(csv_file_path)
    except FileNotFoundError:
        return jsonify({'error': f'File not found at {csv_file_path}'}), 400

    # Get recommendations using the function
    top_k = data.get('top_k', 5)  # You can also get this from the request if needed
    recommendations = get_symptom_recommendations_tfidf(df, input_symptoms, top_k)
    
    if recommendations:
        return jsonify({'recommendations': recommendations})
    else:
        return jsonify({'message': 'No recommendations found'}), 404

if __name__ == '__main__':
    app.run(debug=True)