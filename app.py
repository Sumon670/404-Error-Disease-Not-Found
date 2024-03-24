from flask import Flask, request,render_template, redirect,session, flash, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
import bcrypt
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import re
from pdfminer.high_level import extract_text
import nltk  
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
import tempfile
import joblib
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)
app.secret_key = 'secret_key'


filename = 'diabetes-prediction-model.pkl'
classifier = pickle.load(open(filename, 'rb'))

filename_cancer = 'cancer-prediction-model.pkl'
classifier_cancer = pickle.load(open(filename_cancer, 'rb'))


filename_heart = 'heart-disease-prediction-model.pkl'
classifier_heart = pickle.load(open(filename_heart,'rb'))

# filename_kidney = 'kidney_model.pkl'
classifier_kidney = joblib.load('kidney_model.pkl')
classifier_liver = joblib.load('liver-prediction-model.pkl')

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(100), unique=True)
    password = db.Column(db.String(100))

    def __init__(self,email,password,name):
        self.name = name
        self.email = email
        self.password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    def check_password(self,password):
        return bcrypt.checkpw(password.encode('utf-8'),self.password.encode('utf-8'))

with app.app_context():
    db.create_all()

def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        pdf_file.save(tmp_file.name)
        return extract_text(tmp_file.name)

# Preprocess the text
def preprocess_text(text):
    text = re.sub(r'\[[0-9]*\]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# Calculate sentence similarity
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = set()
    words1 = [word.lower() for word in sent1 if word not in stopwords]
    words2 = [word.lower() for word in sent2 if word not in stopwords]
    all_words = list(set(words1 + words2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    
    for word in words1:
        vector1[all_words.index(word)] += 1
    
    for word in words2:
        vector2[all_words.index(word)] += 1
    
    return cosine_similarity([vector1], [vector2])[0,0]

# Generate summary
def generate_summary(text, top_n=5):
    sentences = sent_tokenize(text)
    clean_sentences = [word_tokenize(sentence) for sentence in sentences]
    stop_words = set(stopwords.words('english'))
    
    sentence_similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i == j:
                continue
            sentence_similarity_matrix[i][j] = sentence_similarity(clean_sentences[i], clean_sentences[j], stop_words)
    
    scores = np.sum(sentence_similarity_matrix, axis=1)
    ranked_sentences = [sentence for _, sentence in sorted(zip(scores, sentences), reverse=True)[:top_n]]
    
    return ranked_sentences

@app.route('/')
def index():
    return render_template('index.html')
@app.route('/home')
def home():
    return render_template('index.html')
@app.route('/aboutus')
def aboutus():
    return render_template('aboutus.html')
@app.route('/services')
def services():
    return render_template('services.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')
@app.route('/help')
def help():
    return render_template('help.html')


@app.route('/register',methods=['GET','POST'])
def register():
    if request.method == 'POST':
        # handle request
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']

        new_user = User(name=name,email=email,password=password)
        db.session.add(new_user)
        db.session.commit()
        return redirect('/login')
    
    return render_template('register.html')

@app.route('/login',methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        
        if user and user.check_password(password):
            session['email'] = user.email
            return redirect('/dashboard')
        else:
            flash('Invalid username or password.')
            return render_template('login.html',error='Invalid user')

    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if session['email']:
        user = User.query.filter_by(email=session['email']).first()
        return render_template('dashboard.html',user=user)
    
    return redirect('/login')
@app.route('/dash_about')
def dash_about():
    return render_template('dash_about.html')
@app.route('/dash_contact')
def dash_contact():
    return render_template('dash_contact.html')
@app.route('/dash_help')
def dash_help():
    return render_template('dash_help.html')
@app.route('/diabetes')
def diabetes():
    return render_template('diabetes.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)
        print(my_prediction)
        return render_template('diabetes_result.html', prediction=my_prediction)

@app.route('/cancer')
def cancer():
    return render_template('breast_cancer.html')

@app.route('/predict_b', methods=['POST'])
def predict_b():
    if request.method == 'POST':
        radius_mean = float(request.form['radius_mean'])
        texture_mean = float(request.form['texture_mean'])
        perimeter_mean = float(request.form['perimeter_mean'])
        area_mean = float(request.form['area_mean'])
        smoothness_mean = float(request.form['smoothness_mean'])
        compactness_mean = float(request.form['compactness_mean'])
        concavity_mean = float(request.form['concavity_mean'])
        concave_points_mean = float(request.form['concave points_mean'])
        symmetry_mean = float(request.form['symmetry_mean'])
        
        data = np.array([[radius_mean,texture_mean,perimeter_mean,area_mean, smoothness_mean,compactness_mean,concavity_mean,concave_points_mean,symmetry_mean]])
        my_prediction_cancer = classifier_cancer.predict(data)
        print (my_prediction_cancer)
        return render_template('breast_cancer_result.html', prediction=my_prediction_cancer)
    
@app.route('/heart')
def heart():
    return render_template('hearts.html')

@app.route('/predict_h', methods=['POST'])
def predict_h():
    if request.method == 'POST':
        Age = int(request.form['Age'])
        Sex = int(request.form['Sex'])
        ChestPainType = int(request.form['ChestPainType'])
        RestingBP = int(request.form['RestingBP'])
        Cholesterol = int(request.form['Cholesterol'])
        FastingBS = int(request.form['FastingBS'])
        RestingECG = int(request.form['RestingECG'])
        MaxHR = int(request.form['MaxHR'])
        ExerciseAngina= int(request.form['ExerciseAngina'])
        Oldpeak = int(request.form['Oldpeak'])
        ST_Slope = int(request.form['ST_Slope'])

        data = np.array([[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]])
        my_prediction_heart = classifier_heart.predict(data)
        
        print (my_prediction_heart)
        return render_template('hearts_result.html', prediction=my_prediction_heart)

@app.route('/kidney')
def kidney():
    return render_template('kidney.html')

@app.route('/predict_k',methods=['POST'])
def predict_k():
    if request.method == 'POST':
        bp = request.form['bp']
        sg = request.form['sg']
        al = request.form['al']
        su = request.form['su']
        rbc = request.form['rbc']
        pc= request.form['pc']
        data = np.array([[bp,sg,al,su,rbc,pc]])
        my_prediction_kidney = classifier_kidney.predict(data)
        print (my_prediction_kidney)
        return render_template('kidney_result.html', prediction=my_prediction_kidney)

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/predict_l',methods=['POST'])
def predict_l():
    if request.method == 'POST':
        Age = request.form['Age']
        Gender = request.form['Gender']
        Total_Bilirubin = request.form['Total_Bilirubin']
        Direct_Bilirubin = request.form['Direct_Bilirubin']
        Alkaline_Phosphotase = request.form['Alkaline_Phosphotase']
        Alamine_Aminotransferase= request.form['Alamine_Aminotransferase']
        Aspartate_Aminotransferase= request.form['Aspartate_Aminotransferase']
        Total_Protiens= request.form['Total_Protiens']
        Albumin= request.form['Albumin']
        data = np.array([[Age,Gender,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin]])
        my_prediction_liver = classifier_liver.predict(data)
        print (my_prediction_liver)
        return render_template('liver_result.html', prediction=my_prediction_liver)

@app.route('/report_sum')
def report_sum():
    return render_template('report_sum.html')

@app.route('/summary', methods=['POST'])
def summary():
    if request.method == 'POST':
        pdf_file = request.files['pdf_file']
        text = extract_text_from_pdf(pdf_file)
        preprocessed_text = preprocess_text(text)
        summarized_sentences = generate_summary(preprocessed_text)
        return render_template('report_sum.html', summarized_text=summarized_sentences)
    return render_template('report_sum.html', summarized_text=None)

@app.route('/doc_recom')
def doc_recom():
    return render_template('doc_recom.html')

def load_doctor_data():
    with open('models\data\doctors_dataset.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

doctor_data = load_doctor_data()

# Function to filter doctors based on city, state, and disease
def filter_doctors(city, state, disease):
    filtered_doctors = []
    for doctor in doctor_data:
        if doctor['City'].lower() == city.lower() and doctor['State'] == state and doctor['Disease'] == disease:
            # Include only the required information: name, state, city, and contact info
            filtered_doctors.append({
                'Name': doctor['Name'],
                'State': doctor['State'],
                'City': doctor['City'],
                'Contact Info': doctor['Contact Number']
            })
    return filtered_doctors

# Route to handle form submission and display recommended doctors
@app.route('/recommend', methods=['POST'])
def recommend_doctors():
    city = request.form['city']
    state = request.form['state']
    disease = request.form['disease']
    doctors = filter_doctors(city, state, disease)
    return render_template('doctors.html', doctors=doctors)

@app.route('/medicine')
def medicine():
    return render_template('medicine.html')

sym_dis = pd.read_csv('models\data\symtoms_df.csv')
precautions = pd.read_csv('models\data\precautions_df.csv')
workout = pd.read_csv('models\data\workout_df.csv')
description = pd.read_csv('models\data\description.csv')
medications = pd.read_csv('models\data\medications.csv')
diets = pd.read_csv('models\data\diets.csv')
svc = pickle.load(open("svc.pkl",'rb'))

def helper(dis):
    descr =  description[description['Disease'] == dis]['Description']
    descr = " ".join([w for w in descr])

    pre = precautions[precautions['Disease'] == dis][['Precaution_1','Precaution_2','Precaution_3','Precaution_4']]
    pre = [col for col in pre.values]

    med = medications[medications['Disease']==dis]['Medication']
    med = [med for med in med.values]

    diet = diets[diets['Disease']==dis]['Diet']
    diet = [diet for diet in diet.values]

    wrk = workout[workout['disease']==dis]['workout']

    return descr, pre, med, diet, wrk

symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal  Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

def pred_value(p_sym):
    inp_vec = np.zeros(len(symptoms_dict))
    for item in p_sym:
        inp_vec[symptoms_dict[item]] = 1
    return diseases_list[svc.predict([inp_vec])[0]]

@app.route('/predict_sym', methods = ['POST','GET'])
def predict_sym():
    if request.method == 'POST':
        symptoms = request.form.get('symptoms')
        user_sym = [ s.strip() for s in symptoms.split(",")]
        user_sym = [s.strip("[]' ") for s in user_sym]
        predicted_dis = pred_value(user_sym)
        descr, pre, med, diet, wrk = helper(predicted_dis)
        ch_pre = []
        for i in pre[0]:
            ch_pre.append(i)
        ch_med = []
        for i in med:
            ch_med.append(i)
        ch_diet = []
        for i in diet:
            ch_diet.append(i)
        return render_template('medicine.html',predicted_dis=predicted_dis,dis_desc=descr,dis_pre=ch_pre,dis_med=ch_med,dis_wrk=wrk, dis_diet=ch_diet)

@app.route('/chatbot')
def chatbot():
    return render_template('chat.html')
@app.route("/get", methods=["GET","POST"])
def chat():
    msg = request.form['msg']
    input = msg
    return get_chat_response(input)

def get_chat_response(text):

    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

    # Let's chat for 5 lines
    for step in range(5):
        # encode the new user input, add the eos_token and return a tensor in Pytorch
        new_user_input_ids = tokenizer.encode(str(text) + tokenizer.eos_token, return_tensors='pt')

        # append the new user input tokens to the chat history
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

        # generated a response while limiting the total chat history to 1000 tokens, 
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # pretty print last ouput tokens from bot
        return tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
@app.route('/logout')
def logout():
    session.pop('email',None)
    return redirect('/login')

if __name__ == '__main__':
    app.run(debug=True)

