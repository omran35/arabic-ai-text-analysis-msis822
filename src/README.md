# تنفيذ المشروع في ملفات Python

النوت بوك يستورد الدوال من هذا المجلد ويعرض نتائجها. لا يعرّف نسخًا أخرى منها.
عند فتح الدفتر في Colab، خلية الإعداد تجلب المستودع وتثبت المكتبات وتجهز الاستيراد تلقائيًا.

| الملف | الوظيفة |
|---|---|
| data_preparation.py | تحميل الأجزاء، فحص الأعمدة والجودة، الدمج والتصنيف، ربط المصادر، حذف التكرار وحجز الأقسام |
| data_preprocessing.py | تطبيع العربية، كلمات وقف NLTK، ISRI ومعالجة قسم التدريب |
| analysis.py | الإحصاءات الوصفية وn-grams والكلمات الوظيفية والترقيم والمصطلحات |
| visualization.py | عرض العربية ورسم النتائج وحفظ الأشكال |
| utils.py | إعدادات المصدر والمجلدات وحفظ الجداول والتقرير وحزمة النتائج |
| __init__.py | تعريف src كحزمة Python |

بعد خلية الإعداد في الدفتر:

```python
from src.data_preparation import (
    load_raw_datasets, get_dataset_columns, check_column_consistency, combine_datasets,
)

df1, df2, df3 = load_raw_datasets(project=PROJECT)
print(get_dataset_columns(df1))
print(check_column_consistency(df1, df2, df3))
wide = combine_datasets(df1, df2, df3)
```

التحميل الافتراضي من نسخة Hugging Face المثبتة. ويمكن استخدام
`load_raw_datasets(input_dir='data/raw')` لقراءة ملفات CSV الثلاثة
`by_polishing.csv` و`from_title.csv` و`from_title_and_content.csv` عند توفرها.
تبقى معلومات طريقة التوليد ومعرف صف المصدر محفوظة بعد الدمج.

```python
from src.data_preprocessing import get_arabic_stopwords, preprocess_text

stop_words = get_arabic_stopwords(PROJECT / 'data/external/nltk_data', download=True)
versions = preprocess_text('إِنَّ هَذِهِ دِرَاسَةٌ عَرَبِيَّةٌ.', stop_words)
print(versions)
```

الاستيراد وحده لا ينزل بيانات أو يشغل التحليل. التنزيل يحدث عند استدعاء التحميل أو
طلب قائمة الوقف بـ`download=True`. تمرر الدوال مدخلاتها صراحة ولا تعتمد على متغيرات النوت بوك.
المراحل 3 و4 و5 تبقى خططًا مستقبلية.
