import pandas as pd

def run_pipeline(input_path, output_path):
    # 1. تحميل البيانات
    df = pd.read_csv(input_path)
    
    # 2. تنظيف البيانات (Cleaning)
    # حذف القيم المفقودة وتحويل الأنواع
    df.dropna(inplace=True)
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    # 3. هندسة الميزات (Feature Engineering)
    # حساب السن والمدة بالدقائق كما فعلنا في التحليل السابق
    df['age'] = 2026 - df['member_birth_year']
    df['duration_min'] = df['duration_sec'] / 60
    
    # 4. تصفية البيانات (Filtering)
    # إزالة القيم الشاذة (Outliers) التي اكتشفناها في الرسم البياني
    df = df[df['age'] <= 80] 
    
    # 5. حفظ البيانات النهائية
    df.to_csv(output_path, index=False)
    print(f"✅ Pipeline finished! Clean data saved to: {output_path}")

if __name__ == "__main__":
    run_pipeline('fordgobike-tripdata.csv', 'cleaned_bike_data.csv')