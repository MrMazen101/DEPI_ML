import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV, ShuffleSplit

# استدعاء ملف الـ visuals الخاص بيك
import visuals as vs

# -----------------------------------------------------------
# 1. Model Training & Caching
# -----------------------------------------------------------
@st.cache_resource
def load_and_train_model():
    # Load dataset
    data = pd.read_csv('housing.csv')
    prices = data['MEDV']
    features = data.drop('MEDV', axis=1)
    
    # Setup GridSearch
    cv_sets = ShuffleSplit(n_splits=10, test_size=0.20, random_state=0)
    regressor = DecisionTreeRegressor(random_state=0)
    params = {'max_depth': list(range(1, 11))}
    scoring_fnc = make_scorer(r2_score)
    
    grid = GridSearchCV(regressor, params, scoring=scoring_fnc, cv=cv_sets)
    grid = grid.fit(features, prices)
    
    return grid.best_estimator_, features, prices

# -----------------------------------------------------------
# 2. GUI Layout - Sidebar Predictions
# -----------------------------------------------------------
st.set_page_config(page_title="Boston Housing Predictor", page_icon="🏡", layout="wide")

st.title("🏡 Boston Housing Price Predictor")
st.markdown("Estimate the selling price of a home based on key neighborhood features.")

# تحميل الموديل والبيانات
model, features, prices = load_and_train_model()

# Sidebar for user inputs
st.sidebar.header("🏠 Input Property Features")
rm = st.sidebar.slider("Average number of rooms (RM)", min_value=1.0, max_value=10.0, value=5.0, step=0.1)
lstat = st.sidebar.slider("Neighborhood poverty level % (LSTAT)", min_value=0.0, max_value=100.0, value=17.0, step=0.1)
ptratio = st.sidebar.slider("Student-teacher ratio (PTRATIO)", min_value=1.0, max_value=50.0, value=15.0, step=0.1)

# تجهيز البيانات للتوقع
client_data = [[rm, lstat, ptratio]]
predicted_price = model.predict(client_data)[0]

st.sidebar.write("---")
st.sidebar.success(f"### Predicted Price: \n## **${predicted_price:,.2f}**")

# -----------------------------------------------------------
# 3. Visualizations & Testing using visuals.py
# -----------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.header("📊 Complexity Curve")
    st.write("Visualize the **Bias-Variance Tradeoff** across different tree depths.")
    
    if st.button("Generate Complexity Curve", type="primary"):
        with st.spinner("Calculating performance..."):
            # استخدام الدالة من ملف visuals.py
            fig = vs.ModelComplexity(features, prices)
            st.pyplot(fig)

with col2:
    st.header("🔄 Predict Trials")
    st.write("Test model stability by predicting 10 times with different data splits.")
    
    if st.button("Run 10 Prediction Trials", type="primary"):
        with st.spinner("Running trials..."):
            
            # بنعمل دالة صغيرة عشان ندرب الموديل بنفس العمق الأفضل (max_depth)
            optimal_depth = model.get_params()['max_depth']
            def fast_fitter(X, y):
                reg = DecisionTreeRegressor(max_depth=optimal_depth, random_state=0)
                reg.fit(X, y)
                return reg
            
            # استخدام الدالة من ملف visuals.py
            trial_prices = vs.PredictTrials(features, prices, fast_fitter, client_data)
            
            # عرض النتائج في جدول أنيق
            trials_df = pd.DataFrame({
                "Trial": range(1, 11),
                "Predicted Price": [f"${p:,.2f}" for p in trial_prices]
            })
            st.dataframe(trials_df, hide_index=True)
            
            # عرض الفرق (Range)
            price_range = max(trial_prices) - min(trial_prices)
            st.warning(f"**Range in prices:** ${price_range:,.2f}")