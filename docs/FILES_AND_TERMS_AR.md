# دليل الملفات والمصطلحات

## أين يوجد كل شيء؟

| المسار | ما الذي أجده فيه؟ |
|---|---|
| notebooks/MSIS822_Phase1_Phase2_Arabic_AI.ipynb | الكود الحالي والنتائج والشرح والملاحظات |
| notebooks/03 و04 و05 | ملفات ملاحظات للخطوات المستقبلية فقط |
| data/raw/ | ينشأ عند التشغيل ويحفظ الأجزاء الأصلية بصيغة Parquet |
| data/processed/phase1_corpus.parquet | النصوص الصالحة بعد ضبط الجودة والتكرار، مع القسم ومعلومات المصدر |
| data/processed/phase2_training.parquet وcsv | نصوص التدريب بنسخها المعالجة وإحصاءاتها الوصفية |
| data/processed/partition_manifest.csv | خريطة ثابتة تربط معرف النص والمصدر بقسم train/validation/test |
| data/external/arabic_stopwords_normalized.txt | قائمة الوقف بعد تطبيق التطبيع نفسه |
| reports/phase1_phase2_findings.md | التقرير المرحلي بالأرقام الفعلية وحدودها |
| reports/figures/ | 10 صور للمقارنات وسحابات الكلمات |
| reports/tables/phase1_audit.json | الأعداد العامة لعمليات ضبط الجودة |
| reports/tables/phase1_*.csv | فحص الأعمدة والجودة والتوزيع والتقسيم وحالات العزل |
| reports/tables/phase2_*.csv | أمثلة المعالجة والإحصاءات وتكرار الكلمات والعبارات |
| docs/run_metadata.json | بيانات بيئة التشغيل المحلي والبذرة ونسخة المصدر |
| docs/phase1_review_ar.md | سبب تطوير النسخة الأولى والفرق بينها وبين الدفتر المعتمد |
| requirements.txt | مكتبات تشغيل متوافقة |
| requirements-tested.txt | إصدارات مكتبات الاختبار المحلي المحددة |
| src/data_preparation.py | دوال مستقلة لمعالجة العربية وكلمات الوقف وISRI |
| src/visualization.py | أدوات عرض العربية وحفظ الأشكال |
| src/utils.py | إعدادات البيانات والمجلدات المشتركة |
| docs/REQUIREMENTS_AUDIT_AR.md | مطابقة متطلبات المقرر بالمنجز والخطوات اللاحقة |
| models/ | مكان النماذج بعد تنفيذ المرحلة الرابعة |

ملفات data/raw وdata/processed والكاش كبيرة وقابلة لإعادة البناء؛ لا أحتاج رفعها إلى GitHub.
قد لا تظهر محليًا قبل تشغيل الدفتر. نتائج التقارير والجداول الصغيرة موجودة في المستودع للمراجعة.

## أهم أعمدة البيانات

| العمود | المعنى |
|---|---|
| text_raw | النص كما جاء من المصدر |
| text_normalized | النص بعد التطبيع وحصر النسخة المعالجة في الحروف العربية |
| text_no_stopwords | النسخة المطبعة بعد حذف كلمات الوقف |
| text_stemmed | نسخة الكلمات بعد ISRI |
| label / class_name | 0 Human أو 1 AI |
| text_id | بصمة ثابتة للنص الكامل بعد توحيد Unicode والمسافات |
| group_id | عائلة المصدر التي تبقى داخل قسم واحد |
| partition | train أو validation أو test |
| generation_methods / generators | مصدر التوليد؛ معلومات تتبع وليست مدخلات تصنيف |
| source_rows / occurrences | الصفوف التي ظهر فيها النص وعدد مرات ظهوره |

## قراءة إحصاءات المرحلة الثانية

| الاسم | المعنى |
|---|---|
| word_count | عدد الكلمات العربية بعد التطبيع |
| avg_word_length | متوسط عدد حروف الكلمة |
| sentence_count / avg_sentence_length | عدد الجمل ومتوسط كلماتها وفق فصل تقريبي بالترقيم |
| ttr | عدد الكلمات المختلفة ÷ عدد الكلمات |
| ttr_first_50 | TTR لأول 50 كلمة للنصوص المؤهلة فقط |
| function_words_per_100_words | نسبة الكلمات المطابقة لقائمة الوقف من الكلمات |
| punctuation_per_100_chars | علامات من المجموعة المحددة في الكود لكل 100 حرف أصلي |
| diacritics_per_100_chars | علامات التشكيل المحددة لكل 100 حرف أصلي |
| retained_word_fraction | نسبة الكلمات التي بقيت بعد حذف كلمات الوقف |

count عدد القيم المتاحة، mean المتوسط، median الوسيط، std الانحراف المعياري.
NaN تعني أن القيمة غير متاحة لهذه الحالة؛ لا أعاملها كصفر بلا تفسير.
