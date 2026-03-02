import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
# ==========================================
# 2. Custom Feature Engineering Transformer
# ==========================================
class BankFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        # مفيش إعدادات مبدئية محتاجينها هنا، بس بنجهز الـ init
        pass
        
    def fit(self, X, y=None):
        # الـ Feature Engineering بتاعنا مش محتاج يتعلم حاجة من الداتا (زي الـ Mean مثلا)
        # هو مجرد عمليات حسابية، فهنرجع الكلاس زي ما هو
        return self
    
    def transform(self, X):
        # بناخد نسخة من الداتا عشان مانعدلش على الأصلية بالغلط
        X_new = X.copy()
        
        # 💡 الميزة 1: نسبة الرصيد للمرتب (Balance to Salary Ratio)
        # العميل اللي رصيده عالي جداً مقارنة بمرتبه غالباً سلوكه مختلف
        # ضفنا 1e-6 عشان نتجنب القسمة على صفر
        X_new['Balance_Salary_Ratio'] = X_new['Balance'] / (X_new['EstimatedSalary'] + 1e-6)
        
        # 💡 الميزة 2: تفاعل العمر مع مدة البقاء (Age and Tenure Interaction)
        # تقييم استقرار العميل (عميل كبير وقديم vs شاب وجديد)
        X_new['Tenure_Age_Ratio'] = X_new['Tenure'] / X_new['Age']
        
        # 💡 الميزة 3: سكور النشاط (Activity Score)
        # دمجنا وجود الكريدت كارد مع كونه عضو نشط
        X_new['Activity_Score'] = X_new['IsActiveMember'] + X_new['HasCrCard']
        
        # 💡 الميزة 4: تصنيف الرصيد (Is Zero Balance)
        # البنوك بتهتم جداً بالعميل اللي رصيده صفر لأنه أقرب للـ Churn
        X_new['Is_Zero_Balance'] = (X_new['Balance'] == 0).astype(int)
        
        return X_new

# تجربة سريعة للـ Transformer (للتأكد إنه شغال)
#f_dummy = pd.DataFrame({'Balance': [1000, 0], 'EstimatedSalary': [5000, 2000], 'Tenure': [5, 2], 'Age': [30, 25], 'IsActiveMember': [1, 0], 'HasCrCard': [1, 1]})
#engineer = BankFeatureEngineer()
#print(engineer.transform(f_dummy))