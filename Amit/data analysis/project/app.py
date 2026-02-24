import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1. إعدادات الصفحة والبراند
st.set_page_config(page_title="Ford GoBike Analysis", layout="wide")
# اللون الأزرق الرسمي للبراند
brand_color = '#007db8' 
sns.set_theme(style="whitegrid")

# 2. تحميل البيانات (مع التأكد من وجودها)
@st.cache_data
def load_data():
    return pd.read_csv('cleaned_bike_data.csv')

df = load_data()

# القائمة الجانبية للتنقل بين الـ 15 رسمة
st.sidebar.title("dashboard sections for ford gobike data analysis📊")
section = st.sidebar.radio("choose a section to explore:", 
    ["1. Univariate (8 Plots)", "2. Bivariate (4 Plots)", "3. Multivariate (3 Plots)"])

# --- القسم الأول: Univariate Exploration (8 رسومات) ---
if section == "1. Univariate (8 Plots)":
    st.title("📊 Univariate Exploration")
    st.info("there are 8 univariate plots that will be added here to explore the distribution of each variable in the dataset. For now, we will show 4 of them as an example.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # 1. Duration in Minutes
        st.subheader("Trip Duration (Minutes)")
        fig, ax = plt.subplots()
        bins = np.arange(0, 60, 2)
        plt.hist(df['duration_min'], bins=bins, color=brand_color)
        st.pyplot(fig)
        
        # 2. Age Distribution
        st.subheader("User Age Distribution")
        fig, ax = plt.subplots()
        plt.hist(df['age'], bins=20, color=brand_color)
        st.pyplot(fig)

    with col2:
        # 3. User Type
        st.subheader("User Type Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='user_type', color=brand_color, ax=ax)
        st.pyplot(fig)
        
        # 4. Member Gender
        st.subheader("Gender Count")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='member_gender', color=brand_color, ax=ax)
        st.pyplot(fig)

    # كمل باقي الـ 8 رسومات هنا (Days, Hours, Bike Share, etc.) بنفس الطريقة
    st.write("the rest of the univariate plots will be added here following the same structure.")

# --- القسم الثاني: Bivariate Exploration (4 رسومات) ---
elif section == "2. Bivariate (4 Plots)":
    st.title("📈 Bivariate Exploration")
    
    # 9. Age vs User Type
    st.subheader("Age vs. User Type")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.boxplot(data=df, x='user_type', y='age', color=brand_color, ax=ax)
    st.pyplot(fig)
    st.write("**Analysis:** Median age is consistent across user types.")

    # 10. Duration vs Gender
    st.subheader("Trip Duration vs. Gender")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(data=df, x='member_gender', y='duration_min', color=brand_color, inner='quartile', ax=ax)
    plt.ylim(0, 40)
    st.pyplot(fig)

# --- القسم الثالث: Multivariate (3 رسومات) ---
elif section == "3. Multivariate (3 Plots)":
    st.title("🧬 Multivariate & Correlation")
    
    # 13, 14, 15 Correlation & Heatmaps
    st.subheader("Correlation Between Age & Duration")
    fig, ax = plt.subplots()
    # استخدام ألوان vlag المتناسقة مع البراند
    sns.heatmap(df[['age', 'duration_min', 'duration_hour']].corr(), annot=True, cmap='vlag', center=0, ax=ax)
    st.pyplot(fig)
    st.success("analysis: Age has a weak positive correlation with trip duration.")
    
    st.balloons()