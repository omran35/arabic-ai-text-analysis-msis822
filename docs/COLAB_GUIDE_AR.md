# تشغيل المشروع في Google Colab

1. افتح الدفتر من زر **Open in Colab** في README.
2. اختر بيئة CPU؛ لا تحتاج GPU للمرحلتين الحاليتين.
3. شغل خلية الإعداد الأولى. Colab يفتح ملف ipynb فقط من GitHub، لذلك تجلب الخلية
   المستودع إلى `/content/arabic-ai-text-detection1` عند غيابه وتثبت `requirements.txt`.
4. شغل خلية الاستيراد ثم بقية الخلايا بالترتيب، أو اختر **Runtime > Run all**.
5. ستظهر الجداول والرسوم تحت خلايا الاستدعاء، وتُحفظ داخل
   `/content/arabic-ai-text-detection1/MSIS822_Arabic_AI_Phases_1_2`.
6. لتنزيل ZIP النتائج، أزل التعليق عن سطري `files.download` في آخر خلية.

لا تحتاج PyCharm أو مجلد `.venv` أو `uv.lock` أو رفع `src` يدويًا.
عند تعديل دالة عدّل ملفها في `src`؛ النوت بوك يستدعيها بالاستيراد مثل:

```python
from src.data_preparation import load_raw_datasets, combine_datasets

df1, df2, df3 = load_raw_datasets(project=PROJECT)
wide = combine_datasets(df1, df2, df3)
```

خلية الإعداد تعيد استخدام نسخة المشروع الموجودة وتحافظ على تعديلات الجلسة.
لجلب تحديثات GitHub افتح جلسة Colab جديدة بعد حفظ النتائج المهمة؛ إعادة تشغيل الخلية
في جلسة فيها نسخة سابقة لا تنفذ `git pull` تلقائيًا.

للتشغيل محليًا افتح الدفتر من مجلد `notebooks` أو جذر المستودع؛ الخلية تكتشف الجذر
وتستخدم ملفات `src` المحلية. ملفات النتائج داخل جلسة Colab مؤقتة، فاحفظ ما تحتاجه قبل انتهائها.
