from flask import Flask, render_template, request
import pickle, numpy as np

app = Flask(__name__)

with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get feature values from the form
        Gender = float(request.form['Gender'])
        Married = float(request.form['Married'])
        Dependents = float(request.form['Dependents'])
        Education = float(request.form['Education'])
        Self_Employed = float(request.form['Self_Employed'])
        ApplicantIncome = float(request.form['ApplicantIncome'])
        CoapplicantIncome = float(request.form['CoapplicantIncome'])
        LoanAmount = float(request.form['LoanAmount'])
        Loan_Amount_Term = float(request.form['Loan_Amount_Term'])
        Credit_History = float(request.form['Credit_History'])
        Property_Area = float(request.form['Property_Area'])
        
        # Add more features as needed

        # Create a numpy array with the input values
        input_features = np.array([[Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]])

        # Make prediction using the model
        prediction = model.predict(input_features)

        prediction_str = "Approved" if prediction[0] == 1 else "Not Approved"

        return render_template('index.html', prediction=prediction_str)
    else:
        return "Invalid request method"




if __name__ == '__main__':''
    app.run(debug=True)
