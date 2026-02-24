pip install streamlit
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 1. إعدادات الصفحة
st.set_page_config(page_title="Ford GoBike Dashboard", page_icon="🚲", layout="wide")

st.title("🚲 Ford GoBike Data Exploration")
st.write("hi! Welcome to the Ford GoBike data dashboard. Here you can explore the cleaned dataset and gain insights about bike usage patterns.")
st.markdown("---")

# 2. قراءة الداتا النظيفة
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_bike_data.csv')
    return df

df = load_data()

# 3. عرض عينة من البيانات
st.subheader("take a look at a sample of the cleaned dataset")
st.dataframe(df.head())
st.markdown("---")

# 4. أول رسمة تفاعلية (أنواع المستخدمين)
st.subheader("Distribution of User Types between Subscribers and Customers")
fig, ax = plt.subplots(figsize=(8, 4))
base_color = sns.color_palette()[0]
sns.countplot(data=df, x='user_type', color=base_color, ax=ax)
st.pyplot(fig)