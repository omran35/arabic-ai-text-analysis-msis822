"""تحميل البيانات وفحص جودتها وربط مصادرها وتقسيمها قبل التحليل."""
from pathlib import Path
import hashlib
import os
import tempfile

import numpy as np
import pandas as pd

from .utils import DATASET_ID, REVISION, SEED, METHODS, TEXT_COLUMNS, LABELS
# Preserve imports used by earlier versions of the project.
from .data_preprocessing import (
    canonical, normalize_arabic, normalized_stopwords, get_arabic_stopwords,
    remove_stopwords, stem_token, stem_text, preprocess_text,
)


def load_raw_datasets(input_dir=None, *, project=None, cache_dir=None):
    """أعيد الأجزاء الثلاثة بالترتيب: CSV محلي اختياري أو نسخة Hugging Face المثبتة.

    عند تمرير project أحفظ نسخ Parquet الأصلية فيه. لا أغير الملفات المدخلة.
    """
    if input_dir is not None:
        parts = tuple(pd.read_csv(Path(input_dir) / f'{method}.csv') for method in METHODS)
    else:
        from datasets import load_dataset
        if cache_dir is None and os.name == 'nt':
            cache_dir = Path(tempfile.gettempdir()) / 'm822-hf'
        elif cache_dir is None and project is not None:
            cache_dir = Path(project) / 'data/external/hf_cache'
        dataset = load_dataset(DATASET_ID, revision=REVISION,
                               cache_dir=str(cache_dir) if cache_dir is not None else None)
        parts = tuple(dataset[method].to_pandas() for method in METHODS)
    check_column_consistency(*parts)
    if project is not None:
        raw_dir = Path(project) / 'data/raw'
        raw_dir.mkdir(parents=True, exist_ok=True)
        for method, part in zip(METHODS, parts):
            part.to_parquet(raw_dir / f'{method}.parquet', index=False)
    return parts


def get_dataset_columns(frame):
    """أعيد أسماء الأعمدة لعرضها داخل النوت بوك."""
    return frame.columns.tolist()


def check_column_consistency(df1, df2, df3):
    """أتحقق من تطابق أسماء الأعمدة ومن وجود الأعمدة النصية الخمسة المطلوبة."""
    expected = set(df1.columns)
    for method, frame in zip(METHODS, (df1, df2, df3)):
        if set(frame.columns) != expected or not set(TEXT_COLUMNS).issubset(frame.columns):
            raise ValueError(f'Unexpected columns in {method}: {list(frame.columns)}')
    return True


def combine_datasets(df1, df2, df3):
    """أدمج الأجزاء بعد إضافة طريقة التوليد ومعرف صف المصدر دون تعديل المدخلات."""
    check_column_consistency(df1, df2, df3)
    frames = []
    for method, frame in zip(METHODS, (df1, df2, df3)):
        part = frame.copy()
        part['generation_method'] = method
        part['source_row'] = [f'{method}:{i}' for i in range(len(part))]
        frames.append(part)
    return pd.concat(frames, ignore_index=True)

def inspect_raw_datasets(df1, df2, df3):
    """أحسب أنواع الأعمدة والنقص والتكرار قبل أي تنظيف."""
    check_column_consistency(df1, df2, df3)
    schema_rows, quality_rows = [], []
    # ملاحظة: أكرر الفحص نفسه على طرق التوليد الثلاث دون خلطها بالتقسيم الإحصائي.
    for method, part in zip(METHODS, (df1, df2, df3)):
        for col in TEXT_COLUMNS:
            values = part[col]
            schema_rows.append({'method': method, 'column': col, 'dtype': str(values.dtype)})
            # ملاحظة: كل سطر في تقرير الجودة يخص عمودًا نصيًا داخل إحدى طرق التوليد.
            quality_rows.append({
                'method': method, 'column': col, 'rows': len(part),
                'missing': int(values.isna().sum()),
                'blank_strings': int(values.map(lambda x: isinstance(x,str) and not x.strip()).sum()),
                'non_string_non_null': int(values.map(lambda x: not isinstance(x,str) and pd.notna(x)).sum()),
                'exact_duplicate_non_null_cells': int(values.dropna().duplicated().sum()),
                'exact_duplicate_wide_rows': int(part.duplicated().sum())})

    schema = pd.DataFrame(schema_rows)
    quality = pd.DataFrame(quality_rows)
    return schema, quality

def prepare_labeled_texts(wide):
    """أحوّل الأعمدة إلى نص وتصنيف وأعزل النقص وغير العربي وتعارض التصنيف."""
    wide = wide.copy()
    wide['source_group'] = wide.apply(
        lambda r: hashlib.sha256(canonical(r['original_abstract']).encode()).hexdigest()
        if isinstance(r['original_abstract'], str) and r['original_abstract'].strip()
        else 'missing-original:'+r['source_row'], axis=1)
    # ملاحظة: التحويل من خمسة أعمدة نصية إلى عمود نص واحد مع عمود يبين المصدر.
    long = wide.melt(id_vars=['source_row','source_group','generation_method'],
                     value_vars=TEXT_COLUMNS, var_name='source_column', value_name='text_raw')
    # ملاحظة: التصنيف يأتي من عمود المصدر، ولا نستنتجه من محتوى النص.
    long['label'] = long['source_column'].ne('original_abstract').astype(int)
    long['class_name'] = long['label'].map({0:'Human',1:'AI'})
    long['generator'] = long['source_column'].str.replace('_generated_abstract','',regex=False)
    long.loc[long['label'].eq(0), 'generator'] = 'human'
    long['is_valid_string'] = long['text_raw'].map(lambda x: isinstance(x,str) and bool(x.strip()))
    invalid = long.loc[~long['is_valid_string']].copy()
    valid = long.loc[long['is_valid_string']].copy()
    valid['text_key'] = valid['text_raw'].map(canonical)
    valid['text_normalized'] = valid['text_raw'].map(normalize_arabic)
    no_arabic = valid.loc[valid['text_normalized'].eq('')].copy()
    valid = valid.loc[valid['text_normalized'].ne('')].copy()
    # ملاحظة: أبحث عن النص المطبع الذي ظهر بتصنيفين متعارضين لعزله.
    conflict_keys = valid.groupby('text_normalized')['label'].nunique()
    conflict_keys = set(conflict_keys[conflict_keys.gt(1)].index)
    conflicts = valid.loc[valid['text_normalized'].isin(conflict_keys)].copy()
    valid = valid.loc[~valid['text_normalized'].isin(conflict_keys)].copy()
    return long, valid, invalid, no_arabic, conflicts

def deduplicate_corpus(valid, long):
    """أربط عائلات المصدر المشتركة وأحذف التكرار مع حفظ معلوماته."""
    valid = valid.copy()
    # ملاحظة الطالب: حذف التكرار وربط المصدر
    # Union-find connects source families that share a normalized text.
    # ملاحظة: قائمة ربط عائلات المصدر؛ ليست نموذج تعلم آلي.
    parent = {g:g for g in valid['source_group'].unique()}
    def find(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    # ملاحظة: إذا اشتركت عائلتان في نص مطبع أربطهما بمجموعة واحدة.
    def union(a,b):
        a,b = find(a),find(b)
        if a != b:
            parent[max(a,b)] = min(a,b)
    first_group = {}
    for text, group in zip(valid['text_normalized'], valid['source_group']):
        if text in first_group:
            union(group, first_group[text])
        else:
            first_group[text] = group
    valid['group_id'] = valid['source_group'].map(find)
    def joined(values):
        return '|'.join(sorted(set(values)))
    # ملاحظة: أجمع معلومات المصدر قبل حذف الصفوف المتكررة حتى لا أفقدها.
    provenance = valid.groupby('text_key', sort=False).agg(
        generation_methods=('generation_method',joined), generators=('generator',joined),
        source_rows=('source_row',joined), occurrences=('source_row','size')).reset_index()
    # ملاحظة: نسخة واحدة لكل نص كامل بعد توحيد Unicode والمسافات.
    corpus = valid.drop_duplicates('text_key').drop(columns=['generation_method','generator','source_row'])
    corpus = corpus.merge(provenance,on='text_key',validate='one_to_one').reset_index(drop=True)
    corpus['text_id'] = corpus['text_key'].map(lambda s:hashlib.sha256(s.encode()).hexdigest())
    assert corpus['text_id'].is_unique
    assert corpus.groupby('text_normalized')['group_id'].nunique().max() == 1
    class_counts = pd.DataFrame({
        'raw_text_slots': long['class_name'].value_counts(),
        'valid_nonconflicting_occurrences': valid['class_name'].value_counts(),
        'unique_exact_texts': corpus['class_name'].value_counts()}).reindex(LABELS).fillna(0).astype(int)
    class_counts['unique_percent'] = (100*class_counts['unique_exact_texts']/len(corpus)).round(2)
    return corpus, class_counts

def reserve_partitions(corpus, *, seed=SEED):
    """أحجز المجموعات ببذرة ثابتة وأفحص عدم تسرب النصوص والمصادر."""
    corpus = corpus.copy()
    # ملاحظة الطالب: الحجز المبكر للبيانات
    groups = np.array(sorted(corpus['group_id'].unique()))
    # ملاحظة: تقليب ثابت لعائلات المصدر باستخدام البذرة المحددة.
    np.random.default_rng(seed).shuffle(groups)
    n_train, n_val = int(.70*len(groups)), int(.15*len(groups))
    assert min(n_train,n_val,len(groups)-n_train-n_val) > 0
    split_map = {g:'train' for g in groups[:n_train]}
    split_map.update({g:'validation' for g in groups[n_train:n_train+n_val]})
    split_map.update({g:'test' for g in groups[n_train+n_val:]})
    # ملاحظة: كل النصوص التابعة للمجموعة نفسها تحصل على القسم نفسه.
    corpus['partition'] = corpus['group_id'].map(split_map)
    # ملاحظة: فحص يمنع وجود المجموعة نفسها أو النص المطبع نفسه في أكثر من قسم.
    assert corpus.groupby('group_id')['partition'].nunique().max() == 1
    assert corpus.groupby('text_normalized')['partition'].nunique().max() == 1
    assert corpus.groupby('partition')['label'].nunique().eq(2).all()
    partition_counts = pd.crosstab(corpus['partition'],corpus['class_name']).reindex(
        index=['train','validation','test'],columns=LABELS,fill_value=0)
    partition_counts['total'] = partition_counts.sum(axis=1)
    partition_counts['percent_of_texts'] = (100*partition_counts['total']/len(corpus)).round(2)
    return corpus, partition_counts

def build_quality_audit(wide, long, valid, invalid, no_arabic, conflicts, corpus):
    """أجمع أعداد ضبط الجودة للتقرير والتدقيق."""
    audit = {'dataset_id':DATASET_ID,'revision':REVISION,'source_rows':len(wide),
             'raw_text_slots':len(long),'invalid_slots':len(invalid),'no_arabic_rows':len(no_arabic),
             'conflicting_rows':len(conflicts),'exact_duplicate_rows_removed':len(valid)-len(corpus),
             'unique_texts':len(corpus),'source_families_after_linking':int(corpus['group_id'].nunique())}
    return audit
