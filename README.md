# Student Performance Prediction 🎓

This is an end-to-end Machine Learning project designed to predict a student's Mathematics score based on several demographic and socioeconomic variables. The project is built with a modular architecture to ensure scalability and ease of deployment.

---

## 📌 Project Overview
The application uses a regression-based approach to analyze how factors like parental education, lunch status, and test preparation impact student outcomes. It includes a complete pipeline from raw data to a web-based prediction interface.

### Key Features
* **Modular Codebase:** Organized into ingestion, transformation, and training components.
* **Automated Model Selection:** Evaluates multiple algorithms (Linear Regression, Random Forest, XGBoost, etc.) and selects the best performer.
* **Custom Exception Handling:** Centralized logging and error management for robust production use.
* **Web UI:** Interactive Flask interface for real-time predictions.

---

## 🛠️ Tech Stack
* **Python 3.8+**
* **Machine Learning:** Scikit-learn, XGBoost, CatBoost
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Deployment:** Flask

---

## 📂 Project Structure
```text
├── artifacts/             # Stored serialized models (.pkl) and processed data
├── notebook/              # Exploratory Data Analysis (EDA) and model experimentation
├── src/                   # Source code directory
│   ├── components/        # Data Ingestion, Data Transformation, Model Trainer
│   ├── pipeline/          # Training and Prediction scripts
│   ├── logger.py          # Script for custom logging
│   └── exception.py       # Script for custom exception handling
├── templates/             # HTML templates for the Flask web application
├── app.py                 # Flask server entry point
├── requirements.txt       # List of required Python packages
└── setup.py               # Package metadata and installation script
