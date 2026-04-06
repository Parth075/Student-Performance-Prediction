import joblib
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# --- Load Models ---
# Load the pre-trained ensemble model and the scaler
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("Error: model.pkl or scaler.pkl not found.")
    # In a real app, you might want to handle this more gracefully
except Exception as e:
    print(f"Error loading models: {e}")

# --- Define Feature Columns (MUST MATCH YOUR TRAINING DATA) ---

# These are the numerical columns that need scaling
NUMERICAL_COLS = [
    'age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours',
    'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating'
]

# These are the original categorical columns before one-hot encoding
CATEGORICAL_COLS = [
    'gender', 'part_time_job', 'diet_quality', 'parental_education_level',
    'internet_quality', 'extracurricular_participation'
]

# !! CRITICAL !!
# This list MUST be the exact order of columns your model was trained on.
# I am creating a plausible list based on your feature names.
# You MUST update this to match the columns from your `X_train` after preprocessing.
MODEL_COLUMNS = ['age', 'study_hours_per_day', 'social_media_hours', 'netflix_hours', 'attendance_percentage', 'sleep_hours', 'exercise_frequency', 'mental_health_rating', 'gender_Male', 'gender_Other', 'part_time_job_Yes', 'diet_quality_Good', 'diet_quality_Poor', 'parental_education_level_High School', 'parental_education_level_Master', 'internet_quality_Good', 'internet_quality_Poor', 'extracurricular_participation_Yes']

# These are used to populate the dropdowns in the HTML form.
# Ensure these match the values in your data.
CATEGORICAL_OPTIONS = {
    'gender': ['Male', 'Female'],
    'part_time_job': ['Yes', 'No'],
    'diet_quality': ['Good', 'Average', 'Poor'],
    'parental_education_level': ["Bachelor's Degree", "Master's Degree", "PhD", "High School", "No Formal Education"],
    'internet_quality': ['Good', 'Average', 'Poor'],
    'extracurricular_participation': ['Yes', 'No']
}

# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    return render_template('index.html', 
                           categorical_options=CATEGORICAL_OPTIONS, 
                           prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles the form submission and returns a prediction."""
    try:
        # 1. Collect form data and convert to correct types
        form_data = {
            # Numerical
            'age': int(request.form['age']),
            'study_hours_per_day': float(request.form['study_hours_per_day']),
            'social_media_hours': float(request.form['social_media_hours']),
            'netflix_hours': float(request.form['netflix_hours']),
            'attendance_percentage': float(request.form['attendance_percentage']),
            'sleep_hours': float(request.form['sleep_hours']),
            'exercise_frequency': float(request.form['exercise_frequency']),
            'mental_health_rating': float(request.form['mental_health_rating']),
            
            # Categorical
            'gender': request.form['gender'],
            'part_time_job': request.form['part_time_job'],
            'diet_quality': request.form['diet_quality'],
            'parental_education_level': request.form['parental_education_level'],
            'internet_quality': request.form['internet_quality'],
            'extracurricular_participation': request.form['extracurricular_participation']
        }

        # 2. Create a DataFrame from the form data
        input_df = pd.DataFrame([form_data])

        # 3. Preprocessing: Scale numerical features
        # The scaler was fitted on the numerical columns only
        input_df[NUMERICAL_COLS] = scaler.transform(input_df[NUMERICAL_COLS])

        # 4. Preprocessing: One-hot encode categorical features
        # This creates new columns (e.g., 'gender_Male', 'gender_Female')
        input_encoded = pd.get_dummies(input_df, columns=CATEGORICAL_COLS)

        # 5. Preprocessing: Align columns with the trained model
        # This is the most important step!
        # It adds any missing one-hot columns (with 0) and
        # ensures the order is identical to the one used for training.
        input_final = input_encoded.reindex(columns=MODEL_COLUMNS, fill_value=0)

        # 6. Make prediction
        prediction = model.predict(input_final)
        
        # Format the output
        output_score = round(prediction[0], 2)
        prediction_text = f'Predicted Exam Score: {output_score}'

    except Exception as e:
        prediction_text = f'Error processing input: {e}'
        print(f"Error: {e}") # Log the error for debugging

    # 7. Render the page again, this time with the prediction
    return render_template('index.html', 
                           categorical_options=CATEGORICAL_OPTIONS, 
                           prediction_text=prediction_text)

# --- Run the App ---
if __name__ == '__main__':
    app.run(debug=True)