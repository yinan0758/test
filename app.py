import streamlit as st
import numpy as np
import xgboost as xgb

# 登录状态初始化
if "logged_in" not in st.session_state: 
    st.session_state.logged_in  = False

# 登录验证函数
def check_login(username, password):
    # 示例验证逻辑（生产环境需加密存储）
    return username == "admin" and password == "250224"

# 主界面显示图片
def show_main_content():
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

# 登录界面
if not st.session_state.logged_in: 
    with st.form("login"): 
        username = st.text_input(" 用户名")
        password = st.text_input(" 密码", type="password")
        submitted = st.form_submit_button(" 登录")
        
        if submitted:
            if check_login(username, password):
                st.session_state.logged_in  = True
                st.rerun()   # 强制刷新页面 [5]()
            else:
                st.error(" 认证失败")

# 登录成功后显示内容
if st.session_state.logged_in: 
    show_main_content()
