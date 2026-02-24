
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Ford GoBike Dashboard", page_icon="🚲", layout="wide")

st.title("🚲 Ford GoBike Data Exploration")
st.write("أهلاً بيك في الداشبورد الخاصة بتحليل بيانات نظام مشاركة الدراجات!")
st.markdown("---")

# 2. قراءة الداتا النظيفة
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_bike_data.csv')
    return df

df = load_data()

# 3. عرض عينة من البيانات
st.subheader("نظرة سريعة على البيانات (أول 5 صفوف)")
st.dataframe(df.head())
st.markdown("---")

# 4. أول رسمة تفاعلية (أنواع المستخدمين)
st.subheader("توزيع المستخدمين حسب النوع (Subscriber vs Customer)")
fig, ax = plt.subplots(figsize=(8, 4))
base_color = sns.color_palette()[0]
sns.countplot(data=df, x='user_type', color=base_color, ax=ax)
st.pyplot(fig)