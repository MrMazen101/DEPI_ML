import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# ==========================================
# 1. Configuration Management
# ==========================================
class Config:
    # 🔹 إعدادات عامة
    SEED = 42  # تثبيت العشوائية لضمان نفس النتائج في كل مرة
    N_SPLITS = 5  # عدد طيات الـ Cross-Validation
    
    # 🔹 مسارات الملفات (تأكد من تعديلها حسب مسار الداتا في مسابقتك)
    TRAIN_PATH = "Bank_Churn.csv" # أو مسار Kaggle: '/kaggle/input/.../Bank_Churn.csv'
    TEST_PATH = "Bank_Churn.csv" # أو مسار Kaggle: '/kaggle/input/.../Bank_Churn_Test.csv'
    SUBMISSION_PATH = "submission.csv"
    
    # 🔹 أسماء الأعمدة (بناءً على وصف الداتا في المسابقة)
    TARGET = "Exited" # أو 'churn' حسب اسم العمود في الداتا
    ID_COL = "CustomerId" # عمود الـ ID اللي مش هنستخدمه في التدريب

    DROP_COLS = ["RowNumber", "CustomerId", "Surname"] # الأعمدة اللي مش هتدخل في التدريب (لو موجودة)   
    
    # 🔹 تقسيم الميزات (Features)
    NUM_FEATURES = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary',  'NumOfProducts'] # الميزات الرقمية اللي هتدخل في الـ Pipeline
    CAT_FEATURES = ['Geography', 'Gender']
    BIN_FEATURES = ['HasCrCard', 'IsActiveMember'] # ميزات ثنائية (0 أو 1)