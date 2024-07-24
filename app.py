import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import joblib

# تحميل البيانات
# data = pd.read_csv('/content/sampled_dataa.csv')

# تعريف معالج البيانات
class PrepProcessor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        # تجهيز المعوضات للقيم الناقصة
        self.imputers = {
            'oldbalanceOrg': SimpleImputer(strategy='mean'),
            'newbalanceOrig': SimpleImputer(strategy='mean'),
            'oldbalanceDest': SimpleImputer(strategy='mean'),
            'newbalanceDest': SimpleImputer(strategy='mean'),
            'isFlaggedFraud': SimpleImputer(strategy='mean')
        }
        
        for column, imputer in self.imputers.items():
            imputer.fit(X[[column]])
        
        # تجهيز المحولات للأعمدة الفئوية
        self.label_encoders = {
            'nameOrig': LabelEncoder(),
            'nameDest': LabelEncoder()
        }
        
        for column, le in self.label_encoders.items():
            le.fit(X[column])
        
        return self

    def transform(self, X, y=None):
        # تعويض القيم الناقصة
        for column, imputer in self.imputers.items():
            X[column] = imputer.transform(X[[column]])
        
        # تحويل الأعمدة الفئوية إلى أرقام
        for column, le in self.label_encoders.items():
            X[column] = le.transform(X[column])
        
        return X

# تحديد أعمدة البيانات
columns = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 'newbalanceOrig', 
           'nameDest', 'oldbalanceDest', 'newbalanceDest', 'isFlaggedFraud']
model = joblib.load('model.joblib')

st.title("تطبيق كشف الاحتيال في المدفوعات")

# إدخال بيانات المستخدم
step = st.number_input('الخطوة', min_value=0)
# typee = st.selectbox('النوع', data['type'].unique())
amount = st.number_input('المبلغ', min_value=0.0)
nameOrig = st.text_input('اسم المرسل', 'C123456')
oldbalanceOrg = st.number_input('الرصيد القديم للمرسل', min_value=0.0)
newbalanceOrig = st.number_input('الرصيد الجديد للمرسل', min_value=0.0)
nameDest = st.text_input('اسم المستلم', 'C654321')
oldbalanceDest = st.number_input('الرصيد القديم للمستلم', min_value=0.0)
newbalanceDest = st.number_input('الرصيد الجديد للمستلم', min_value=0.0)
isFlaggedFraud = st.number_input('Is Flagged Fraud', min_value=0.0)

def predict(): 
    row = np.array([step,amount, nameOrig, oldbalanceOrg, newbalanceOrig, nameDest, oldbalanceDest, newbalanceDest, isFlaggedFraud]) 
    X = pd.DataFrame([row], columns = columns)
    
    # تحضير البيانات باستخدام المعالج
    processor = PrepProcessor()
    processor.fit(data)
    X_transformed = processor.transform(X)
    
    # التنبؤ باستخدام النموذج
    prediction = model.predict(X_transformed)
    
    if prediction[0] == 1: 
        st.success('تم اكتشاف معاملة احتيالية :thumbsup:')
    else: 
        st.error('المعاملة ليست احتيالية :thumbsdown:') 

trigger = st.button('تنبؤ', on_click=predict)
