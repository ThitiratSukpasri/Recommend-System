# Symptom Recommendation API

This project provides a simple Flask API for recommending symptoms based on a user's input using K-Nearest Neighbors (KNN). It is designed to assist in medical data exploration by finding related symptoms from an existing dataset.

## Features

-   Input: List of symptoms
-   Output: List of recommended related symptoms
-   KNN-based recommendation using patient-symptom matrix
-   Supports Thai language and multi-symptom input

## Project Structure


├── app.py          # Main Flask application file
├── data/           # Directory containing the symptom data
│   └── symptom_data.csv  # CSV file with symptom data
├── requirements.txt  # List of Python dependencies
└── README.md       # Project documentation (this file)


## Getting Started

### Installation

1.  Clone this repository:

    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

2.  Install the required Python dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3.  Run the Flask application:

    ```bash
    python app.py
    ```
