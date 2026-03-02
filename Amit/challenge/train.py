import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier

# استدعاء الكلاسات اللي عملناها في الملفات التانية
from config import Config
from pipeline import build_preprocessing_pipeline

# ==========================================
# 4. Cross-Validation & Training Engine
# ==========================================
def train_and_evaluate(train_df, test_df):
    """
    الدالة دي بتدرب الموديل باستخدام Stratified K-Fold وتمنع أي تسريب للبيانات.
    """
    print("🚀 جاري بدء التدريب والتقييم...")
    
    # فصل الميزات (X) عن الهدف (y)
    X = train_df.drop(columns=[Config.TARGET] + Config.DROP_COLS, errors='ignore')
    y = train_df[Config.TARGET]
    
    # تجهيز بيانات الاختبار (بدون عمود الـ ID)
    X_test = test_df.drop(columns=Config.DROP_COLS, errors='ignore')
    
    # مصفوفات فاضية عشان نحفظ فيها التوقعات
    oof_predictions = np.zeros(len(train_df))
    test_predictions = np.zeros(len(test_df))
    fold_scores = []
    
    # تعريف الـ K-Fold (مع التوزيع العادل للفئات - Stratified)
    skf = StratifiedKFold(n_splits=Config.N_SPLITS, shuffle=True, random_state=Config.SEED)
    
    # اللوب الأساسي للتدريب
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
        
        # 1. بناء خط الإنتاج (Pipeline)
        pipeline = build_preprocessing_pipeline(Config.NUM_FEATURES, Config.CAT_FEATURES)
        
        # 2. معالجة البيانات (Fit على الـ Train فقط، و Transform للـ Val والـ Test)
        # هنا بنضمن 100% إن مفيش Data Leakage
        X_train_processed = pipeline.fit_transform(X_train)
        X_val_processed = pipeline.transform(X_val)
        X_test_processed = pipeline.transform(X_test)
        
        # 3. تعريف الموديل (LightGBM)
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=5,
            random_state=Config.SEED,
            class_weight='balanced', # مهم جداً لو الداتا فيها Imbalance
            verbose=-1
        )
        
        # 4. تدريب الموديل
        model.fit(X_train_processed, y_train)
        
        # 5. التوقع لبيانات الـ Validation (بناخد الاحتمالات للفئة 1)
        val_preds = model.predict_proba(X_val_processed)[:, 1]
        oof_predictions[val_idx] = val_preds
        
        # حساب الـ ROC AUC للـ Fold ده
        fold_auc = roc_auc_score(y_val, val_preds)
        fold_scores.append(fold_auc)
        print(f"✅ Fold {fold + 1} | ROC AUC: {fold_auc:.5f}")
        
        # 6. التوقع لبيانات الـ Test (وناخد المتوسط لكل الـ Folds)
        test_preds = model.predict_proba(X_test_processed)[:, 1]
        test_predictions += test_preds / Config.N_SPLITS
        
    # ==========================================
    # 📊 التقييم النهائي
    # ==========================================
    print("-" * 40)
    print(f"🎯 متوسط الـ ROC AUC لكل الطيات: {np.mean(fold_scores):.5f} ± {np.std(fold_scores):.5f}")
    
    # تقييم الـ Out-Of-Fold الشامل
    oof_score = roc_auc_score(y, oof_predictions)
    print(f"🏆 النتيجة النهائية الشاملة (OOF ROC AUC): {oof_score:.5f}")
    print("-" * 40)
    
    return test_predictions, model, pipeline

# للتجربة:
# train_data = pd.read_csv(Config.TRAIN_PATH)
# test_data = pd.read_csv(Config.TEST_PATH)
# final_test_preds, last_model, last_pipeline = train_and_evaluate(train_data, test_data)