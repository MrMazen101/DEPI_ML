import pandas as pd
import shap
import matplotlib.pyplot as plt

# استدعاء الشغل بتاعنا من الملفات التانية
from config import Config
from train import train_and_evaluate

# ==========================================
# 5. Main Execution & Explainability
# ==========================================

def generate_submission(test_df, predictions, output_path):
    """
    بتاخد توقعات الموديل (الاحتمالات) وبتعمل ملف الـ Submission النهائي بشكل مبرمج.
    """
    submission = pd.DataFrame({
        Config.ID_COL: test_df[Config.ID_COL],
        Config.TARGET: predictions # دي الـ Continuous Probabilities (0.0 to 1.0)
    })
    
    submission.to_csv(output_path, index=False)
    print(f"✅ تم حفظ ملف التوقعات بنجاح في: {output_path}")

def explain_model_with_shap(model, pipeline, sample_data):
    """
    بترسم SHAP Summary Plot عشان تشرح للجنة التحكيم الموديل بيفكر إزاي.
    """
    print("🔍 جاري تحليل الموديل باستخدام SHAP...")
    
    # تحضير عينة من الداتا (بدون عمود الهدف أو الأعمدة المرفوضة) عشان ندخلها للموديل
    X_sample = sample_data.drop(columns=[Config.TARGET] + Config.DROP_COLS, errors='ignore')
    
    # لازم نعدي الداتا على الـ Pipeline الأول عشان الموديل يفهمها
    X_processed = pipeline.transform(X_sample)
    
    # تعريف الـ SHAP Explainer للموديلات الشجرية (زي LightGBM)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_processed)
    
    # رسم الـ Summary Plot
    plt.figure(figsize=(10, 6))
    # لو الموديل Binary Classification، بناخد الـ shap_values للـ class رقم 1
    shap.summary_plot(shap_values[1] if isinstance(shap_values, list) else shap_values, 
                      X_processed, 
                      plot_type="dot", 
                      show=False)
    
    plt.title("SHAP Feature Importance - Customer Churn Drivers", fontsize=14)
    plt.tight_layout()
    plt.savefig("shap_summary.png") # بنحفظ الرسمة عشان تعرضها في النوت بوك
    print("📊 تم حفظ رسمة SHAP بنجاح كـ 'shap_summary.png'")
    plt.show()

if __name__ == "__main__":
    print("🚀 بدء تشغيل نظام توقع الـ Churn الاحترافي...")
    
    # 1. قراءة البيانات
    try:
        train_df = pd.read_csv(Config.TRAIN_PATH)
        test_df = pd.read_csv(Config.TEST_PATH)
        print("📂 تم تحميل البيانات بنجاح.")
    except FileNotFoundError:
        print("❌ خطأ: ملفات البيانات مش موجودة. اتأكد من المسارات في ملف config.py")
        exit()

    # 2. تدريب الموديل والتقييم (Cross-Validation)
    # الدالة دي هترجعلنا التوقعات النهائية، وآخر موديل اتدرب، وآخر Pipeline عشان نستخدمهم في الـ SHAP
    final_predictions, trained_model, fitted_pipeline = train_and_evaluate(train_df, test_df)

    # 3. إنشاء ملف التسليم الأوتوماتيكي
    generate_submission(test_df, final_predictions, Config.SUBMISSION_PATH)

    # 4. تفسير الموديل (استخدام عينة من بيانات التدريب للتفسير)
    # بناخد عينة عشوائية صغيرة (مثلا 1000 صف) عشان الرسمة تطلع بسرعة
    sample_for_shap = train_df.sample(n=1000, random_state=Config.SEED)
    explain_model_with_shap(trained_model, fitted_pipeline, sample_for_shap)
    
    print("🎉 اكتمل المشروع بنجاح! ملف الـ submission.csv جاهز للرفع.")