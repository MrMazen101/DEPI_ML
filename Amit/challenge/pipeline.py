import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# لو انت مقسم الملفات، هتحتاج تعمل Import للكلاسات اللي فاتت
# from config import Config
# from features import BankFeatureEngineer

# ==========================================
# 3. Data Preprocessing Pipeline
# ==========================================
def build_preprocessing_pipeline(num_features, cat_features):
    """
    بتبني خط إنتاج كامل لمعالجة البيانات بدون أي Data Leakage.
    """
    
    # 🔹 1. خط معالجة المتغيرات الرقمية (Numerical Pipeline)
    # لو في قيم ناقصة بنحط الـ Median، وبعدين بنعمل Scaling عشان الموديلات تكون مستقرة
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # 🔹 2. خط معالجة المتغيرات الفئوية (Categorical Pipeline)
    # لو في قيم ناقصة بنحط القيمة الأكثر تكراراً، وبعدين بنحول الكلمات لأرقام (One-Hot)
    # استخدمنا handle_unknown='ignore' عشان لو ظهرت فئة جديدة في الـ test ميعملش Error
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])

    # 🔹 3. تجميع الخطوط في ColumnTransformer
    # ده بيحدد كل عمود هيمشي في أنهي مسار بالظبط
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ],
        remainder='passthrough' # المتغيرات الثنائية (الـ Binary) هتعدي زي ما هي
    )

    # 🔹 4. بناء الـ Pipeline النهائي (End-to-End)
    # بيبدأ بـ Feature Engineering وبعدين بيدخل على الـ Preprocessor
    # هنحط أسماء الميزات الجديدة اللي عملناها عشان يحصلها Scaling مع الرقمية
    
    # تحديث قائمة الميزات الرقمية لتشمل الميزات اللي إحنا اخترعناها
    engineered_features = ['Balance_Salary_Ratio', 'Tenure_Age_Ratio', 'Activity_Score', 'Is_Zero_Balance']
    all_num_features = num_features + engineered_features

    # تحديث الـ preprocessor باللستة الجديدة
    preprocessor.transformers[0] = ('num', numeric_transformer, all_num_features)

    full_pipeline = Pipeline(steps=[
        ('feature_engineer', BankFeatureEngineer()), # الكلاس بتاعنا
        ('preprocessor', preprocessor)
    ])

    return full_pipeline

# للتجربة السريعة:
# pipeline = build_preprocessing_pipeline(Config.NUM_FEATURES, Config.CAT_FEATURES)
# print("Pipeline is ready to roll! 🚀")