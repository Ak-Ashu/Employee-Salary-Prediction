# 💼 Employee Salary Prediction App
This is a Streamlit web app that predicts whether an employee earns more than 50K or less than or equal to 50K per year based on various demographic and job-related factors. The model is trained using a cleaned dataset and deployed with a user-friendly interface for both single and batch predictions.

📌 Features
🔍 Single Prediction using sidebar input fields.

📂 Batch Prediction by uploading a CSV file.

✅ Model trained using Logistic Regression, saved with joblib.

🧠 Includes preprocessing like label encoding and feature scaling.

📊 Prediction results viewable and downloadable as CSV.

📁 Files Structure
bash
Copy
Edit
📦 Employee-Salary-Prediction/
├── app.py                  # Streamlit app
├── best_model.pkl          # Trained machine learning model
├── scaler.pkl              # Trained scaler (StandardScaler)
├── sample_input.csv        # Sample format for CSV uploads
├── README.md               # Project documentation (this file)
└── requirements.txt        # Required Python packages
⚙️ How to Run
streamlit run Employee_Salary_page.py
🙋‍♂️ Author
Ashish Kumar
📧 ashishak6969@gmail.com
🔍 Model Details
Algorithm: Logistic Regression
Libraries: pandas, scikit-learn, numpy, joblib, streamlit
Target: Predict salary category (<=50K or >50K)
Encoding: Manual Label Encoding for categorical columns
Scaler: StandardScaler for feature normalization
