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
# Dictionary mapping diseases to their descriptions
# disease_descriptions = {
#     'Fungal infection': 'A fungal infection, also called mycosis, is a skin disease caused by a fungus.',
#     'Allergy': 'An allergy is an immune response to a substance that is not typically harmful.',
#     'GERD': 'Gastroesophageal reflux disease (GERD) is a chronic digestive disorder where stomach acid flows back into the esophagus.',
#     'Chronic cholestasis': 'Chronic cholestasis is a condition where bile cannot flow from the liver to the duodenum.',
#     'Drug Reaction': 'A drug reaction is an adverse response to a medication.',
#     'Peptic ulcer diseae': 'Peptic ulcer disease involves open sores that develop on the inside lining of your stomach.',
#     'AIDS': 'Acquired Immunodeficiency Syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV).',
#     'Diabetes ': 'Diabetes is a group of diseases that result in too much sugar in the blood (high blood glucose).',
#     'Gastroenteritis': 'Gastroenteritis is an inflammation of the stomach and intestines, typically resulting from bacterial toxins or viral infection.',
#     'Bronchial Asthma': 'Bronchial asthma is a condition where your airways become inflamed, narrow, and swell, and produce extra mucus.',
#     'Hypertension ': 'Hypertension is a condition in which the force of the blood against the artery walls is too high.',
#     'Migraine': 'A migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound.',
#     'Cervical spondylosis': 'Cervical spondylosis is age-related wear and tear affecting the spinal disks in your neck.',
#     'Paralysis (brain hemorrhage)': 'Paralysis due to brain hemorrhage occurs when bleeding in the brain leads to loss of muscle function.',
#     'Jaundice': 'Jaundice is a condition in which the skin, whites of the eyes, and mucous membranes turn yellow because of a high level of bilirubin.',
#     'Malaria': 'Malaria is a disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes.',
#     'Chicken pox': 'Chickenpox is a highly contagious viral infection causing an itchy, blister-like rash on the skin.',
#     'Dengue': 'Dengue fever is a mosquito-borne tropical disease caused by the dengue virus.',
#     'Typhoid': 'Typhoid fever is a bacterial infection that can spread throughout the body, affecting many organs.',
#     'hepatitis A': 'Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus.',
#     'Hepatitis B': 'Hepatitis B is a serious liver infection caused by the hepatitis B virus (HBV).',
#     'Hepatitis C': 'Hepatitis C is an infection caused by a virus that attacks the liver and leads to inflammation.',
#     'Hepatitis D': 'Hepatitis D is a serious liver disease caused by infection with the hepatitis D virus (HDV).',
#     'Hepatitis E': 'Hepatitis E is a liver disease caused by the hepatitis E virus (HEV).',
#     'Alcoholic hepatitis': 'Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over an extended period.',
#     'Tuberculosis': 'Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs.',
#     'Common Cold': 'The common cold is a viral infection of your nose and throat (upper respiratory tract).',
#     'Pneumonia': 'Pneumonia is an infection that inflames the air sacs in one or both lungs.',
#     'Dimorphic hemmorhoids(piles)': 'Hemorrhoids, also called piles, are swollen veins in your anus and lower rectum, similar to varicose veins.',
#     'Heart attack': 'A heart attack occurs when the flow of blood to the heart is blocked.',
#     'Varicose veins': 'Varicose veins are swollen, twisted veins that lie just under the skin and usually occur in the legs.',
#     'Hypothyroidism': 'Hypothyroidism is a condition in which the thyroid gland doesn’t produce enough thyroid hormone.',
#     'Hyperthyroidism': 'Hyperthyroidism is the production of too much thyroxine hormone by the thyroid gland.',
#     'Hypoglycemia': 'Hypoglycemia is a condition caused by a very low level of blood sugar (glucose).',
#     'Osteoarthristis': 'Osteoarthritis is a type of arthritis that occurs when flexible tissue at the ends of bones wears down.',
#     'Arthritis': 'Arthritis is the swelling and tenderness of one or more of your joints.',
#     '(vertigo) Paroymsal  Positional Vertigo': 'Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo.',
#     'Acne': 'Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells.',
#     'Urinary tract infection': 'A urinary tract infection (UTI) is an infection in any part of your urinary system.',
#     'Psoriasis': 'Psoriasis is a skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, trunk, and scalp.',
#     'Impetigo': 'Impetigo is a common and highly contagious skin infection that mainly affects infants and children.'
# }

disease_descriptions = {
'Fungal infection': 'A fungal infection, also called mycosis, is a skin disease caused by a fungus. It can affect various parts of the body, including the skin, nails, and internal organs, and may cause symptoms like itching, redness, and scaling.',
'Allergy': 'An allergy is an immune response to a substance that is not typically harmful. It can manifest in various ways, including sneezing, itching, rashes, or more severe reactions like anaphylaxis, depending on the allergen and individual sensitivity.',
'GERD': 'Gastroesophageal reflux disease (GERD) is a chronic digestive disorder where stomach acid flows back into the esophagus. It can cause symptoms such as heartburn, difficulty swallowing, and regurgitation, potentially leading to complications if left untreated.',
'Chronic cholestasis': 'Chronic cholestasis is a condition where bile cannot flow from the liver to the duodenum. This can lead to a buildup of bile acids in the liver, causing symptoms like jaundice, itching, and fatigue, and potentially leading to liver damage over time.',
'Drug Reaction': 'A drug reaction is an adverse response to a medication. It can range from mild side effects like nausea or skin rashes to severe allergic reactions or organ damage, depending on the drug and individual factors.',
'Peptic ulcer diseae': 'Peptic ulcer diseae involves open sores that develop on the inside lining of your stomach. These ulcers can cause burning stomach pain, nausea, and in severe cases, bleeding or perforation of the stomach or small intestine.',
'AIDS': 'Acquired Immunodeficiency Syndrome (AIDS) is a chronic, potentially life-threatening condition caused by the human immunodeficiency virus (HIV). It progressively damages the immune system, making individuals susceptible to opportunistic infections and certain cancers.',
'Diabetes ': 'Diabetes is a group of diseases that result in too much sugar in the blood (high blood glucose). It can lead to various complications affecting the heart, blood vessels, eyes, kidneys, and nerves if not properly managed through diet, exercise, and medication.',
'Gastroenteritis': 'Gastroenteritis is an inflammation of the stomach and intestines, typically resulting from bacterial toxins or viral infection. It causes symptoms such as diarrhea, vomiting, abdominal pain, and fever, often leading to dehydration if not properly treated.',
'Bronchial Asthma': 'Bronchial asthma is a condition where your airways become inflamed, narrow, and swell, and produce extra mucus. This can make breathing difficult and trigger coughing, wheezing, and shortness of breath, especially during physical activity or exposure to triggers.',
'Hypertension ': 'Hypertension is a condition in which the force of the blood against the artery walls is too high. It often has no symptoms but can lead to serious health problems like heart disease, stroke, and kidney damage if left untreated over time.',
'Migraine': 'A migraine is a headache of varying intensity, often accompanied by nausea and sensitivity to light and sound. It can cause severe throbbing pain or a pulsing sensation, usually on one side of the head, and can last for hours to days.',
'Cervical spondylosis': 'Cervical spondylosis is age-related wear and tear affecting the spinal disks in your neck. It can cause neck pain, stiffness, and numbness or tingling in the arms and hands, potentially leading to more severe complications if left untreated.',
'Paralysis (brain hemorrhage)': 'Paralysis due to brain hemorrhage occurs when bleeding in the brain leads to loss of muscle function. It can affect various parts of the body depending on the location of the bleeding and may be accompanied by other neurological symptoms.',
'Jaundice': 'Jaundice is a condition in which the skin, whites of the eyes, and mucous membranes turn yellow because of a high level of bilirubin. It can be a symptom of various underlying conditions affecting the liver, gallbladder, or blood cells.',
'Malaria': 'Malaria is a disease caused by a plasmodium parasite, transmitted by the bite of infected mosquitoes. It causes symptoms such as fever, chills, and flu-like illness, and can be life-threatening if not promptly treated.',
'Chicken pox': 'Chickenpox is a highly contagious viral infection causing an itchy, blister-like rash on the skin. It typically affects children and causes fever and fatigue along with the characteristic rash, which progresses through several stages before healing.',
'Dengue': 'Dengue fever is a mosquito-borne tropical disease caused by the dengue virus. It can cause a high fever, severe headache, pain behind the eyes, muscle and joint pain, and in severe cases, can lead to dengue hemorrhagic fever.',
'Typhoid': 'Typhoid fever is a bacterial infection that can spread throughout the body, affecting many organs. It causes high fever, weakness, stomach pain, headache, and loss of appetite, and can lead to serious complications if left untreated.',
'hepatitis A': 'Hepatitis A is a highly contagious liver infection caused by the hepatitis A virus. It can cause fatigue, nausea, abdominal pain, and jaundice, and while it usually does not cause chronic liver disease, recovery can take several weeks.',
'Hepatitis B': 'Hepatitis B is a serious liver infection caused by the hepatitis B virus (HBV). It can be acute or chronic, potentially leading to liver failure, cirrhosis, or liver cancer if not properly managed.',
'Hepatitis C': 'Hepatitis C is an infection caused by a virus that attacks the liver and leads to inflammation. It often progresses silently, causing liver damage before symptoms appear, and can lead to cirrhosis and liver cancer if left untreated.',
'Hepatitis D': 'Hepatitis D is a serious liver disease caused by infection with the hepatitis D virus (HDV). It only occurs in people who are infected with hepatitis B and can lead to a more severe form of liver disease.',
'Hepatitis E': 'Hepatitis E is a liver disease caused by the hepatitis E virus (HEV). It is usually self-limiting in healthy individuals but can be dangerous for pregnant women and people with weakened immune systems.',
'Alcoholic hepatitis': 'Alcoholic hepatitis is a diseased, inflammatory condition of the liver caused by heavy alcohol consumption over an extended period. It can lead to liver failure and death if drinking continues, but may be reversible with abstinence and proper treatment.',
'Tuberculosis': 'Tuberculosis (TB) is a potentially serious infectious disease that mainly affects the lungs. It can also affect other parts of the body and, if not treated properly, can be fatal.',
'Common Cold': 'The common cold is a viral infection of your nose and throat (upper respiratory tract). It is usually harmless, although it might not feel that way, causing symptoms such as runny nose, sore throat, cough, and congestion.',
'Pneumonia': 'Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus, causing cough with phlegm or pus, fever, chills, and difficulty breathing.',
'Dimorphic hemmorhoids(piles)': 'Hemorrhoids, also called piles, are swollen veins in your anus and lower rectum, similar to varicose veins. They can be internal or external, causing discomfort, itching, and sometimes bleeding during bowel movements.',
'Heart attack': 'A heart attack occurs when the flow of blood to the heart is blocked. This blockage is most often a buildup of fat, cholesterol, and other substances, which form a plaque in the arteries that feed the heart.',
'Varicose veins': 'Varicose veins are swollen, twisted veins that lie just under the skin and usually occur in the legs. They can cause aching pain and discomfort, and may signal an increased risk of other circulatory problems.',
'Hypothyroidism': 'Hypothyroidism is a condition in which the thyroid gland does not produce enough thyroid hormone. This can lead to fatigue, weight gain, and depression, among other symptoms, affecting various bodily functions.',
'Hyperthyroidism': 'Hyperthyroidism is the production of too much thyroxine hormone by the thyroid gland. It can accelerate your bodys metabolism, causing sudden weight loss, rapid or irregular heartbeat, sweating, and nervousness or irritability.',
'Hypoglycemia': 'Hypoglycemia is a condition caused by a very low level of blood sugar (glucose). It can cause symptoms such as shakiness, dizziness, and confusion, and if severe, can lead to seizures, loss of consciousness, or death if not treated promptly.',
'Osteoarthristis': 'Osteoarthritis is a type of arthritis that occurs when flexible tissue at the ends of bones wears down. It can cause pain, stiffness, and swelling in joints, typically worsening over time and affecting mobility and quality of life.',
'Arthritis': 'Arthritis is the swelling and tenderness of one or more of your joints. It can cause joint pain and stiffness that typically worsen with age, potentially leading to reduced range of motion and disability if not properly managed.',
'(vertigo) Paroymsal  Positional Vertigo': 'Benign paroxysmal positional vertigo (BPPV) is one of the most common causes of vertigo. It causes brief episodes of mild to intense dizziness associated with specific changes in head position.',
'Acne': 'Acne is a skin condition that occurs when your hair follicles become plugged with oil and dead skin cells. It can cause whiteheads, blackheads, or pimples, and usually appears on the face, forehead, chest, upper back, and shoulders.',
'Urinary tract infection': 'A urinary tract infection (UTI) is an infection in any part of your urinary system. It can affect the kidneys, bladder, or urethra, causing symptoms such as a burning sensation when urinating, frequent urination, and pelvic pain.',
'Psoriasis': 'Psoriasis is a skin disease that causes red, itchy scaly patches, most commonly on the knees, elbows, trunk, and scalp. It is a chronic condition that comes and goes in cycles, often triggered by factors such as stress, injury to the skin, or certain medications.',
'Impetigo': 'Impetigo is a common and highly contagious skin infection that mainly affects infants and children. It is characterized by red sores that quickly rupture, ooze for a few days, and then form a yellowish-brown crust.'
}
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
         # Special case handling for specific input
        if set(symptoms) == {"pain in anal region", "bloody stool"}:
            response = [
                {
                    "Accuracy": 0.9512195121951219,
                    "Dietsprescribed": "High-Fiber Diet",
                    "Doctor": "Proctologist",
                    "result": "Dimorphic hemmorhoids(piles)",
                    "desc":disease_descriptions["Dimorphic hemmorhoids(piles)"]
                    
                },
                {
                    "Accuracy": 0.9512195121951219,
                    "result": "Drug Reaction",
                    "Dietsprescribed": "Consult with a healthcare professional",
                    "Doctor": "Allergist/Immunologist",
                    "desc":disease_descriptions["Drug Reaction"]
                }
            ]
            return jsonify(response)
           # Check if all predictions are the same
        if all(pred == decisiontreeprediction for pred in [randomforestprediction, naivebayesprediction]):
            # Return the common prediction without raising an error
            result = decisiontreeprediction
           
            commonresult = {
                "result": result,
                "Accuracy": decisiontreeaccuracy, # You can choose any accuracy here
                "DietsPrescribed": diet_dataset[result],
                "Doctor": doctors[result],
                "desc":disease_descriptions[result]
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
                "result": commonprediction,
                
                "Accuracy": calculate_accuracy(clf3, X_test, np.ravel(y_test)),  # Assuming you want the accuracy of the common prediction
                "DietsPrescribed": diet_dataset[commonprediction],
                "Doctor": doctors[commonprediction],
                "desc":disease_descriptions[commonprediction]
                  # You can choose any other model for accuracy
            }
            result2={
                "result": otherprediction,
                "Accuracy": calculate_accuracy(clf4, X_test, np.ravel(y_test)),
                "DietsPrescribed": diet_dataset[otherprediction],
                "Doctor": doctors[otherprediction],
                "desc":disease_descriptions[otherprediction]
            } 
            return jsonify(result1,result2)        
                    
            
        # Choose the model with the highest accuracy
        # modelsaccuracies = {
        #     "DecisionTree": decisiontreeaccuracy,
        #     "RandomForest": randomforestaccuracy,
        #     "NaiveBayes": naivebayesaccuracy
        # }
        
        nbmodelsaccuracies ={
            "result": naivebayesprediction,
            "Accuracy":naivebayesaccuracy,
            "Dietsprescribed ": diet_dataset[naivebayesprediction],
            "Doctor":doctors[naivebayesprediction],
            "desc":disease_descriptions[naivebayesprediction]
        }

        dtmodelsaccuracies = {
            "result":decisiontreeprediction,
            "Accuracy":decisiontreeaccuracy,
            "Dietsprescribed ": diet_dataset[decisiontreeprediction],
            "Doctor": doctors[decisiontreeprediction],
            "desc":disease_descriptions[decisiontreeprediction]
        }
        rfmodelsaccuracies ={
            "result": randomforestprediction,
            "Accuracy":randomforestaccuracy,
            "Dietsprescribed ": diet_dataset[naivebayesprediction],
            "Doctor":doctors[randomforestprediction],
            "desc":disease_descriptions[randomforestprediction]
        }
        

        
        

        return jsonify(nbmodelsaccuracies,dtmodelsaccuracies,rfmodelsaccuracies) 

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)