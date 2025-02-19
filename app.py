import streamlit as st
import numpy as np
import xgboost as xgb

def get_model():
    global model
    model = xgb.XGBClassifier()
    model.load_model('model.bin')

def get_results(Fetal_station,
                AOP,
                HPD,
                HSD):
    global model
    try:
        Fetal_station_value = float(Fetal_station)
    except:
        return "Fetal_station value is invalid!"
    try:
        AOP_value = float(AOP)
    except:
        return "AOP value is invalid!"
    try:
        HPD_value = float(HPD)
    except:
        return "HPD value is invalid!"
    try:
        HSD_value = float(HSD)
    except:
        return "HSD value is invalid!"
    
    prdict_X = np.array([Fetal_station_value, AOP_value, HPD_value, HSD_value]).reshape(1, 4)
    mean_value = np.array([1.14457831, 123.77911647, 34.32248996, 21.63453815])
    std_value = np.array([1.38377747, 15.30819286,  9.69121534,  7.97368739])
    prdict_X = (prdict_X - mean_value) / std_value
    y_pred = model.predict_proba(prdict_X)
    result = 'The probabilty is ' + str(float(y_pred[0, 1]))
    return result

get_model()
st.title("Prediction model of delivery mode\n")

Fetal_station = st.number_input('Fetal station')
AOP = st.number_input('angle of progression(AOP)')
HPD = st.number_input('head-perineum distance(HPD)')
HSD = st.number_input('head-symphysis distance(HSD)')

if st.button('***Click to Start Predict***'):
    result = get_results(Fetal_station, AOP, HPD, HSD)
    st.write(result)