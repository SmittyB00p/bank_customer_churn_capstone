from flask import Flask, render_template, request
import pickle
from pipeline.predict_pipeline import CustomData, PredictPipeline
import numpy as np

app = Flask(__name__)

with open('final_models/rf_model.pkl', 'rb') as model:
    rf_model = pickle.load(model)

@app.route("/")
def hello_world():
    return render_template('home.html')

@app.route("/prediction", methods=['POST', 'GET'])
def predict_data():
    if request.method == 'GET':
        return render_template('prediction.html')
    else:
        if request.form.get('gender') == "male":
            gender_male = 1
            gender_female = 0
        else:
            gender_female = 1
            gender_male = 0
        
        if request.form.get('credit-card') == 'yes':
            cc = 1
        else:
            cc = 0

        if request.form.get('active-member') == 'yes':
            membership = 1
        else:
            membership = 0

        if request.form.get('complain') == 'yes':
            complain = 1
        else:
            complain = 0
        
        if request.form.get('geography') == 'germany':
            locale_germany = 1
            locale_france = 0
            locale_spain = 0
        elif request.form.get('geography') == 'spain':
            locale_spain = 1
            locale_germany = 0
            locale_france = 0
        else:
            locale_france = 1
            locale_germany = 0
            locale_spain = 0
        
        if request.form.get('card-type') == 'silver':
            card_type_silver = 1
            card_type_diamond = 0
            card_type_gold = 0
            card_type_plat = 0
        elif request.form.get('card-type') == 'gold':
            card_type_gold = 1
            card_type_diamond = 0
            card_type_plat = 0
            card_type_silver = 0
        elif request.form.get('card-type') == 'diamond':
            card_type_diamond = 1
            card_type_gold = 0
            card_type_plat = 0
            card_type_silver = 0
        else:
            card_type_plat = 1
            card_type_diamond = 0
            card_type_silver = 0
            card_type_gold = 0

        data=CustomData(
            credit_score=int(request.form.get('credit-score')),
            age=int(request.form.get('age')),
            tenure=int(request.form.get('tenure')),
            balance=float(request.form.get('balance')),
            products=int(request.form.get('products')),
            credit_card=int(cc),
            membership_activity=int(membership),
            estimated_salary=float(request.form.get('salary')),
            complain=int(complain),
            satisfaction_score=int(request.form.get('satisfaction-score')),
            points_earned=float(request.form.get('points-earned')),
            locale_france=int(locale_france),
            locale_germany=int(locale_germany),
            locale_spain=int(locale_spain),
            gender_female=int(gender_female),
            gender_male=int(gender_male),
            card_type_diamond=int(card_type_diamond),
            card_type_gold=int(card_type_gold),
            card_type_plat=int(card_type_plat),
            card_type_silver=int(card_type_silver),
        )

        pred_df=data.get_data_as_df()
        
        print(pred_df)
        
        # predict_pipeline=PredictPipeline()
        # results = predict_pipeline.predict(pred_df)
        # print(results)

        results = rf_model.predict(pred_df)
        print(results)
        
        if results[0] == 0:
            result='This customer is NOT likely to exit the bank!'
        else:
            result='This customer IS likely to exit the bank!'

        return render_template('prediction.html', results=result)
    
if __name__=="__main__":
    app.run(host='0.0.0.0', debug=True)