import streamlit as st
import pandas as pd

from model import Model


st.set_page_config(page_title="Dự đoán bệnh tim", page_icon="❤️")

if __name__ == '__main__':
    st.title("DỰ ĐOÁN BỆNH TIM SỬ DỤNG MÔ HÌNH LOGISTIC REGRESSION")
    
    st.sidebar.title("Bài tập lớn nhóm 4 - Khai thác dữ liệu và ứng dụng")

    model = Model()

    st.header("Nhập thông tin người muốn dự đoán:")

    age = st.number_input("Độ tuổi:", 0, 126, 18)
    sex = st.selectbox("Giới tính:", ['Male', 'Female'])
    cp = st.selectbox("Dạng đau ngực:", ['typical angina', 'asymptomatic', 'non-anginal', 'atypical angina'])
    trestbps = st.number_input("Huyết áp khi nghỉ:", 0, 200, 140)
    chol = st.number_input("cholesterol huyết thanh (mg/dl):", 0, 603, 223)
    fbs = st.selectbox("Đường huyết lúc đói > 120 mg/dl:", [True, False])
    thalch = st.number_input("Nhịp tim tối đa đạt được", 0.000000, 300.000000, 140.000000)
    exang = st.selectbox("Đau thắt ngực do tập thể dục", [True, False])
    oldpeak = st.number_input("oldpeak", -2.600000, 6.200000, 0.500000)
    slope = st.selectbox("slope", ['downsloping', 'flat', 'upsloping'])
    ca = st.number_input("số lượng mạch chính (0-3) được nhuộm màu bằng phương pháp soi huỳnh quang", 0, 3, 0)
    thal = st.selectbox("thal", ['fixed defect', 'normal', 'reversable defect'])

    features = pd.DataFrame({
        'age': age,
        'sex': sex,
        'cp': cp,
        'trestbps': trestbps,
        'chol': chol,
        'fbs': fbs,
        'thalch': thalch,
        'exang': exang,
        'oldpeak': oldpeak,
        'slope': slope,
        'ca': ca,
        'thal': thal
    }, index=[0])

    if st.button("Predict"):
        st.header("Thông tin đầu vào:")
        st.dataframe(features)

        st.header("Kết quả")
        prediction = model.serving_pipeline(features)
        if prediction == 1:
            st.error("Khả năng bị bệnh cao")
        else:
            st.success("Khả năng bị bệnh thấp")
            


    