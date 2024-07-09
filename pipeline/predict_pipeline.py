import pandas as pd
import sys
import pickle

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = 'final_models/rf_model.pkl'
            model = load_object(file_path=model_path)
            preds = model.predict(features)
            return preds
        except Exception as e:
            raise e

class CustomData:
    def __init__(
            self,
            credit_score: int,
            age: int,
            tenure: int,
            balance: float,
            products: int,
            credit_card: int,
            membership_activity: int,
            estimated_salary: float,
            complain: int,
            satisfaction_score: int,
            points_earned: float,
            locale_france: int,
            locale_germany: int,
            locale_spain: int,
            gender_female: int,
            gender_male: int,
            card_type_diamond: int,
            card_type_gold: int,
            card_type_plat: int,
            card_type_silver: int
            ):
        self.gender_male = gender_male
        self.gender_female = gender_female
        self.credit_score = credit_score
        self.age = age
        self.tenure = tenure
        self.balance = balance
        self.products = products
        self.credit_card = credit_card
        self.membership_activity = membership_activity
        self.estimated_salary = estimated_salary
        self.complain = complain
        self.satisfaction_score = satisfaction_score
        self.points_earned = points_earned
        self.locale_germany = locale_germany
        self.locale_spain = locale_spain
        self.locale_france = locale_france
        self.card_type_diamond = card_type_diamond
        self.card_type_gold = card_type_gold
        self.card_type_plat = card_type_plat
        self.card_type_silver = card_type_silver

    def get_data_as_df(self):
        try:
            custom_df_input = {
                "CreditScore": [self.credit_score],
                "Age": [self.age],
                "Tenure": [self.tenure],
                "Balance": [self.balance],
                "NumOfProducts": [self.products],
                "HasCrCard": [self.credit_card],
                "IsActiveMember": [self.membership_activity],
                "EstimatedSalary": [self.estimated_salary],
                "Complain": [self.complain],
                "Satisfaction Score": [self.satisfaction_score],
                "Point Earned": [self.points_earned],
                "Geography_France": [self.locale_france],
                "Geography_Germany": [self.locale_germany],
                "Geography_Spain": [self.locale_spain],
                "Gender_Female": [self.gender_female],
                "Gender_Male": [self.gender_male],
                "Card Type_DIAMOND": [self.card_type_diamond],
                "Card Type_GOLD": [self.card_type_gold],
                "Card Type_PLATINUM": [self.card_type_plat],
                "Card Type_SILVER": [self.card_type_silver]
            }

            return pd.DataFrame(custom_df_input)
        
        except Exception as e:
            raise e

# def get_dummies(df):
#     dummies_df = pd.get_dummies(df)
#     return dummies_df
        
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file:
            model = pickle.load(file)
            return model
        
    except Exception as e:
        raise e