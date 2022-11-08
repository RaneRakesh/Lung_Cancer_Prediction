import pickle
import warnings
warnings.filterwarnings('ignore')

def lung_cancer_prediction_Log(features):
    
    pickled_model = pickle.load(open('Lung_Cancer_log_reg.pkl', 'rb'))
    can_predict0 = str(round(list(pickled_model.predict([features]))[0]))
    

    return str("Lung Cancer boolean value using Logistic Regression is: " + can_predict0)

def lung_cancer_prediction_Dec(features):
    
    pickled_model1 = pickle.load(open('Lung_Cancer_dec_tree.pkl', 'rb'))
    can_predict1 = str(round(list(pickled_model1.predict([features]))[0]))
    


    return str("Lung Cancer boolean value using Decision Tree is: " + can_predict1)
               

def lung_cancer_prediction_Random(features):
    
    pickled_model2 = pickle.load(open('Lung_Cancer_ran_for.pkl', 'rb'))
    can_predict2 = str(round(list(pickled_model2.predict([features]))[0]))


    return str("Lung Cancer boolean value using Random Forest is: " + can_predict2)

