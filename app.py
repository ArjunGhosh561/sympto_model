from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS  

app = Flask(__name__)
CORS(app) 

l1 = ['back pain', 'constipation', 'abdominal pain', 'diarrhoea', 'mild fever', 'yellow urine',
      'yellowing of eyes', 'acute liver failure', 'fluid overload', 'swelling of stomach',
      'swelled lymph nodes', 'malaise', 'blurred and distorted vision', 'phlegm', 'throat irritation',
      'redness of eyes', 'sinus pressure', 'runny nose', 'congestion', 'chest pain', 'weakness in limbs',
      'fast heart rate', 'pain during bowel movements', 'pain in anal region', 'bloody stool',
      'irritation in anus', 'neck pain', 'dizziness', 'cramps', 'bruising', 'obesity', 'swollen legs',
      'swollen blood vessels', 'puffy face and eyes', 'enlarged thyroid', 'brittle nails',
      'swollen extremeties', 'excessive hunger', 'extra marital contacts', 'drying and tingling lips',
      'slurred speech', 'knee pain', 'hip joint pain', 'muscle weakness', 'stiff neck', 'swelling joints',
      'movement stiffness', 'spinning movements', 'loss of balance', 'unsteadiness',
      'weakness of one body side', 'loss of smell', 'bladder discomfort', 'foul smell of urine',
      'continuous feel of urine', 'passage of gases', 'internal itching', 'toxic look (typhos)',
      'depression', 'irritability', 'muscle pain', 'altered sensorium', 'red spots over body', 'belly pain',
      'abnormal menstruation', 'dischromic patches', 'watering from eyes', 'increased appetite', 'polyuria',
      'family history', 'mucoid sputum',
      'rusty sputum', 'lack of concentration', 'visual disturbances', 'receiving blood transfusion',
      'receiving unsterile injections', 'coma', 'stomach bleeding', 'distention of abdomen',
      'history of alcohol consumption', 'fluid overload', 'blood in sputum', 'prominent veins on calf',
      'palpitations', 'painful walking', 'pus filled pimples', 'blackheads', 'scurring', 'skin peeling',
      'silver like dusting', 'small dents in nails', 'inflammatory nails', 'blister', 'red sore around nose',
      'yellow crust ooze']

disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction',
           'Peptic ulcer diseae', 'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension',
           ' Migraine', 'Cervical spondylosis',
           'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue', 'Typhoid', 'hepatitis A',
           'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis',
           'Common Cold', 'Pneumonia', 'Dimorphic hemmorhoids(piles)',
           'Heartattack', 'Varicoseveins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthristis',
           'Arthritis', '(vertigo) Paroymsal  Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
           'Impetigo']
doctors = {
    'Fungal infection': 'Dermatologist',
    'Allergy': 'Allergist/Immunologist',
    'GERD': 'Gastroenterologist',
    'Chronic cholestasis': 'Hepatologist',
    'Drug Reaction': 'Allergist/Immunologist',
    'Peptic ulcer diseae': 'Gastroenterologist',
    'AIDS': 'Infectious Disease Specialist',
    'Diabetes': 'Endocrinologist',
    'Gastroenteritis': 'Gastroenterologist',
    'Bronchial Asthma': 'Pulmonologist',
    'Hypertension': 'Cardiologist',
    ' Migraine': 'Neurologist',
    'Cervical spondylosis': 'Orthopedic Surgeon',
    'Paralysis (brain hemorrhage)': 'Neurologist',
    'Jaundice': 'Hepatologist',
    'Malaria': 'Infectious Disease Specialist',
    'Chicken pox': 'Infectious Disease Specialist',
    'Dengue': 'Infectious Disease Specialist',
    'Typhoid': 'Infectious Disease Specialist',
    'hepatitis A': 'Hepatologist',
    'Hepatitis B': 'Hepatologist',
    'Hepatitis C': 'Hepatologist',
    'Hepatitis D': 'Hepatologist',
    'Hepatitis E': 'Hepatologist',
    'Alcoholic hepatitis': 'Hepatologist',
    'Tuberculosis': 'Pulmonologist',
    'Common Cold': 'Internal Medicine Specialist',
    'Pneumonia': 'Pulmonologist',
    'Dimorphic hemmorhoids(piles)': 'Proctologist',
    'Heartattack': 'Cardiologist',
    'Varicoseveins': 'Vascular Surgeon',
    'Hypothyroidism': 'Endocrinologist',
    'Hyperthyroidism': 'Endocrinologist',
    'Hypoglycemia': 'Endocrinologist',
    'Osteoarthristis': 'Rheumatologist',
    'Arthritis': '(vertigo) Paroymsal  Positional Vertigo',
    'Acne': 'Dermatologist',
    'Urinary tract infection': 'Urologist',
    'Psoriasis': 'Dermatologist',
    'Impetigo': 'Dermatologist'
}

diet_dataset = {
    'Fungal infection': 'Balanced Diet',
    'Allergy': 'Elimination Diet',
    'GERD': ['Low-Acid Diet', 'Fiber-rich Foods'],
    'Chronic cholestasis': 'Low-Fat Diet',
    'Drug Reaction': 'Consult with a healthcare professional',
    'Peptic ulcer diseae': 'Avoid spicy and acidic foods',
    'AIDS': 'Nutrient-dense Diet',
    'Diabetes': 'Balanced Diet with controlled carbohydrates',
    'Gastroenteritis': 'BRAT Diet (Bananas, Rice, Applesauce, Toast)',
    'Bronchial Asthma': 'Anti-inflammatory Diet',
    'Hypertension': 'DASH Diet (Dietary Approaches to Stop Hypertension)',
    ' Migraine': 'Migraine Diet (Avoiding trigger foods)',
    'Cervical spondylosis': 'Anti-inflammatory Diet',
    'Paralysis (brain hemorrhage)': 'Balanced Diet with emphasis on antioxidants',
    'Jaundice': 'Low-Fat and Low-Protein Diet',
    'Malaria': 'High-Protein Diet',
    'Chicken pox': 'Soft and Easy-to-Swallow Foods',
    'Dengue': 'Fluid and Nutrient-Rich Diet',
    'Typhoid': 'Bland and Soft Diet',
    'hepatitis A': 'Low-Fat Diet',
    'Hepatitis B': 'Low-Fat Diet',
    'Hepatitis C': 'Low-Fat Diet',
    'Hepatitis D': 'Low-Fat Diet',
    'Hepatitis E': 'Low-Fat Diet',
    'Alcoholic hepatitis': 'Abstain from alcohol, Low-Fat Diet',
    'Tuberculosis': 'High-Calorie and High-Protein Diet',
    'Common Cold': 'Adequate Fluids, Vitamin C-rich Foods',
    'Pneumonia': 'Balanced Diet with Protein',
    'Dimorphic hemmorhoids(piles)': 'High-Fiber Diet',
    'Heartattack': 'Heart-Healthy Diet (Low-Sodium, Low-Fat)',
    'Varicoseveins': 'High-Fiber Diet',
    'Hypothyroidism': 'Iodine-rich Diet',
    'Hyperthyroidism': 'Iodine-restricted Diet',
    'Hypoglycemia': 'Frequent, Balanced Meals',
    'Osteoarthristis': 'Anti-inflammatory Diet',
    'Arthritis': 'Anti-inflammatory Diet',
    '(vertigo) Paroymsal  Positional Vertigo': 'Low-Salt Diet',
    'Acne': 'Low-Glycemic Diet',
    'Urinary tract infection': 'Adequate Fluids, Cranberry Juice',
    'Psoriasis': 'Anti-inflammatory Diet',
    'Impetigo': 'Balanced Diet with emphasis on Vitamins A and C'
}


# Load training data
df = pd.read_csv('Training.csv')
df.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3, 'Drug Reaction': 4,
                          'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7, 'Gastroenteritis': 8,
                          'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12,
                          'Paralysis (brain hemorrhage)': 13, 'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16,
                          'Dengue': 17, 'Typhoid': 18, 'hepatitis A': 19,
                          'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22, 'Hepatitis E': 23,
                          'Alcoholic hepatitis': 24, 'Tuberculosis': 25,
                          'Common Cold': 26, 'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28,
                          'Heart attack': 29, 'Varicose veins': 30, 'Hypothyroidism': 31,
                          'Hyperthyroidism': 32, 'Hypoglycemia': 33, 'Osteoarthristis': 34, 'Arthritis': 35,
                          '(vertigo) Paroymsal  Positional Vertigo': 36, 'Acne': 37,
                          'Urinary tract infection': 38, 'Psoriasis': 39,
                          'Impetigo': 40}}, inplace=True)
X = df[l1]
y = df[["prognosis"]]
np.ravel(y)

# Load testing data
tr = pd.read_csv('Testing.csv')
tr.replace({'prognosis': {'Fungal infection': 0, 'Allergy': 1, 'GERD': 2, 'Chronic cholestasis': 3,
                          'Drug Reaction': 4, 'Peptic ulcer diseae': 5, 'AIDS': 6, 'Diabetes ': 7,
                          'Gastroenteritis': 8, 'Bronchial Asthma': 9, 'Hypertension ': 10,
                          'Migraine': 11, 'Cervical spondylosis': 12, 'Paralysis (brain hemorrhage)': 13,
                          'Jaundice': 14, 'Malaria': 15, 'Chicken pox': 16, 'Dengue': 17, 'Typhoid': 18,
                          'hepatitis A': 19, 'Hepatitis B': 20, 'Hepatitis C': 21, 'Hepatitis D': 22,
                          'Hepatitis E': 23, 'Alcoholic hepatitis': 24, 'Tuberculosis': 25, 'Common Cold': 26,
                          'Pneumonia': 27, 'Dimorphic hemmorhoids(piles)': 28, 'Heart attack': 29,
                          'Varicose veins': 30, 'Hypothyroidism': 31, 'Hyperthyroidism': 32, 'Hypoglycemia': 33,
                          'Osteoarthristis': 34, 'Arthritis': 35, '(vertigo) Paroymsal  Positional Vertigo': 36,
                          'Acne': 37, 'Urinary tract infection': 38, 'Psoriasis': 39, 'Impetigo': 40}},
           inplace=True)
X_test = tr[l1]
y_test = tr[["prognosis"]]
np.ravel(y_test)

# Load trained models
def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

clf3 = load_model('decision_tree_model.pkl')
clf4 = load_model('random_forest_model.pkl')
gnb = load_model('naive_bayes_model.pkl')

def predict_disease(symptoms, model):
    l2 = [0] * len(l1)

    for symptom in symptoms:
        if symptom in l1:
            index = l1.index(symptom)
            l2[index] = 1

    input_test = [l2]
    prediction = model.predict(input_test)
    predicted = prediction[0]

    return disease[predicted]

def calculate_accuracy(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Define the route for receiving symptoms via POST request
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.json['symptoms']
        symptoms = [s.strip() for s in user_input.split(',')]

        # Decision Tree
        print(f'Input Data: {X_test.iloc[0]}') 
        decisiontreeprediction = predict_disease(symptoms, clf3)
        print(f'Decision Tree Prediction: {decisiontreeprediction}')

        decisiontreeaccuracy = calculate_accuracy(clf3, X_test, np.ravel(y_test))

        # Random Forest
        randomforestprediction = predict_disease(symptoms, clf4)
        randomforestaccuracy = calculate_accuracy(clf4, X_test, np.ravel(y_test))
        print(f'Random Forest Prediction: {randomforestprediction}')

        # Naive Bayes
        naivebayesprediction = predict_disease(symptoms, gnb)
        naivebayesaccuracy = calculate_accuracy(gnb, X_test, np.ravel(y_test))
        print(f'Naive Bayes Prediction: {naivebayesprediction}')
           # Check if all predictions are the same
        if all(pred == decisiontreeprediction for pred in [randomforestprediction, naivebayesprediction]):
            # Return the common prediction without raising an error
            result = decisiontreeprediction
           
            commonresult = {
                "result": result,
                "Accuracy": decisiontreeaccuracy, # You can choose any accuracy here
                "DietsPrescribed": diet_dataset[result],
                "Doctor": doctors[result]
                }

            answer=[]
            answer.append(commonresult)
            
            return jsonify(answer)
        elif any(pred == decisiontreeprediction for pred in [randomforestprediction, naivebayesprediction]):
        # Handle the case where at least two models predict the same disease
            commonprediction = decisiontreeprediction  # or randomforestprediction, or naivebayesprediction (they are all the same)
            othermodelpredictions = [pred for pred in [randomforestprediction, naivebayesprediction] if pred != commonprediction]
            otherprediction = None
            if othermodelpredictions:
                otherprediction = othermodelpredictions[0]
                
            result1 = {
                "CommonPrediction": commonprediction,
                
                "Accuracy": calculate_accuracy(clf3, X_test, np.ravel(y_test)),  # Assuming you want the accuracy of the common prediction
                "DietsPrescribed": diet_dataset[commonprediction],
                "Doctor": doctors[commonprediction]
                  # You can choose any other model for accuracy
            }
            result2={
                "OtherPredictions": otherprediction,
                "Accuracy": calculate_accuracy(clf4, X_test, np.ravel(y_test)),
                "DietsPrescribed": diet_dataset[otherprediction],
                "Doctor": doctors[otherprediction]
            } 
            return jsonify(result1,result2)        
                    
            
        # Choose the model with the highest accuracy
        modelsaccuracies = {
            "DecisionTree": decisiontreeaccuracy,
            "RandomForest": randomforestaccuracy,
            "NaiveBayes": naivebayesaccuracy
        }
        nbmodelsaccuracies ={
            "NaiveBayesPred": naivebayesprediction,
            "Accuracy":naivebayesaccuracy,
            "Dietsprescribed ": diet_dataset[naivebayesprediction],
            "Doctor":doctors[naivebayesprediction]
        }

        dtmodelsaccuracies = {
            "DecisionTreePred":decisiontreeprediction,
            "Accuracy":decisiontreeaccuracy,
            "Dietsprescribed ": diet_dataset[decisiontreeprediction],
            "Doctor": doctors[decisiontreeprediction]
        }
        rfmodelsaccuracies ={
            "RandomForestPred": randomforestprediction,
            "Accuracy":randomforestaccuracy,
            "Dietsprescribed ": diet_dataset[naivebayesprediction],
            "Doctor":doctors[randomforestprediction]
        }
        

        
        

        return jsonify(nbmodelsaccuracies,dtmodelsaccuracies,rfmodelsaccuracies) 

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)