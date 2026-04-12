Student Performance Prediction 🎓This is an end-to-end Machine Learning project that predicts a student's performance (Mathematics score) based on various demographic and socioeconomic factors. The project follows a modular coding approach, including data ingestion, transformation, model training, and a web-based deployment interface.📌 Project OverviewThe goal of this project is to understand how the performance of students is affected by variables such as Gender, Ethnicity, Parental level of education, Lunch, and Test preparation courses.Key Features:Predictive Modeling: Uses regression algorithms to predict math scores.Modular Architecture: Separate components for data pipelines (Ingestion, Transformation, Training).Web Interface: A Flask application where users can input student data and get instant results.Robust Error Handling: Custom logging and exception handling for easier debugging.🛠️ Tech StackLanguage: Python 3.8+Machine Learning: Scikit-learn, XGBoost, CatBoostData Analysis: Pandas, NumPy, Matplotlib, SeabornWeb Framework: FlaskEnvironment: Virtualenv / Conda📂 Project StructurePlaintext├── artifacts/             # Saved models (model.pkl) and preprocessors
├── notebook/              # Jupyter notebooks for EDA and Model Training
├── src/                   # Source code
│   ├── components/        # Data Ingestion, Transformation, Model Trainer
│   ├── pipeline/          # Training and Prediction pipelines
│   ├── logger.py          # Logging configuration
│   └── exception.py       # Custom exception handling
├── templates/             # HTML files for Flask web app
├── app.py                 # Flask entry point
├── requirements.txt       # Project dependencies
└── setup.py               # Metadata for package installation
🚀 Getting Started1. Clone the repositoryBashgit clone https://github.com/Parth075/Student-Performance-Prediction.git
cd Student-Performance-Prediction
2. Create and activate a virtual environmentBashpython -m venv venv
# On Windows
venv\Scripts\activate
# On Mac/Linux
source venv/bin/activate
3. Install dependenciesBashpip install -r requirements.txt
4. Run the applicationBashpython app.py
After running, open your browser and go to http://127.0.0.1:5000📊 DatasetThe dataset used is the Student Performance in Exams dataset from Kaggle.Target Variable: math_scoreFeatures: Gender, Race/Ethnicity, Parental level of education, Lunch, Test preparation course, Reading score, Writing score.🧠 Model PerformanceThe project evaluates several models:Linear RegressionLasso/Ridge RegressionDecision Tree / Random ForestXGBoost / CatBoost / AdaBoostThe best performing model (highest $R^2$ score) is automatically selected and saved for the prediction pipeline.
