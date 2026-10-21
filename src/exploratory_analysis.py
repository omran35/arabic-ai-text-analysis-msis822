"""إحصاءات وصفية للكلمات والجمل والترقيم؛ تستخدم بيانات التدريب فقط."""
from collections import Counter
import re

import numpy as np
import pandas as pd

from .data_preprocessing import normalize_arabic, DIACRITICS
from .utils import LABELS

SENTENCE_BOUNDARY = re.compile(r'[.!?؟؛;\n]+')
PUNCTUATION = set('.,،;؛:!?؟"«»()[]{}…-–—')
# ملاحظة: هذه إحصاءات وصفية للنص، وليست مصفوفة خصائص تدريب في هذه المرحلة.
def linguistic_stats(row, stop_words):
    raw, normalized = row['text_raw'], row['text_normalized']
    words = normalized.split()
    n = len(words)
    sentence_tokens = [normalize_arabic(s).split() for s in SENTENCE_BOUNDARY.split(raw)]
    sentence_lengths = [len(s) for s in sentence_tokens if s]
    return {
        'word_count':n, 'avg_word_length':sum(map(len,words))/n if n else np.nan,
        'sentence_count':len(sentence_lengths),
        'avg_sentence_length':float(np.mean(sentence_lengths)) if sentence_lengths else np.nan,
        'ttr':len(set(words))/n if n else np.nan,
        'ttr_first_50':len(set(words[:50]))/50 if n>=50 else np.nan,
        'function_words_per_100_words':100*sum(w in stop_words for w in words)/n if n else np.nan,
        'punctuation_per_100_chars':100*sum(c in PUNCTUATION for c in raw)/max(len(raw),1),
        'diacritics_per_100_chars':100*len(DIACRITICS.findall(raw))/max(len(raw),1),
        'retained_word_fraction':len(row['text_no_stopwords'].split())/n if n else np.nan}

def summarize_linguistics(eda, stop_words):
    """أعيد نصوص التدريب بإحصاءاتها وملخص الفئات وملخص طرق التوليد."""
    stats = pd.DataFrame([linguistic_stats(r, stop_words) for _,r in eda.iterrows()])
    eda = pd.concat([eda,stats],axis=1)
    METRICS = list(stats.columns)
    # ملاحظة: أجمع الإحصاءات حسب الفئة دون الجمع بين Human وAI في متوسط واحد.
    summary = eda.groupby('class_name')[METRICS].agg(['count','mean','median','std']).reindex(LABELS)

    method_summary = eda.groupby(['class_name','generation_methods'])[
        ['word_count','avg_sentence_length','ttr']].agg(['count','mean'])
    return eda, METRICS, summary, method_summary

def ngram_counts(texts, n=1, *, stop_words):
    counts, opportunities = Counter(),0
    for text in texts:
        words = text.split()
        # ملاحظة: عدد المواضع الممكنة هو مقام معدل تكرار n-gram.
        opportunities += max(len(words)-n+1,0)
        for i in range(len(words)-n+1):
            gram=tuple(words[i:i+n])
            if not all(w in stop_words for w in gram):
                counts[' '.join(gram)]+=1
    return counts,opportunities


def analyze_ngrams(eda, stop_words):
    """أحسب تكرار الكلمات والعبارات داخل كل نص ومعدلاتها داخل كل فئة."""
    results = {}
    for n in (1, 2, 3):
        for class_name in LABELS:
            counts, total = ngram_counts(
                eda.loc[eda['class_name'].eq(class_name), 'text_normalized'],
                n, stop_words=stop_words)
            top = pd.DataFrame(counts.most_common(15), columns=['ngram', 'count'])
            top['per_10000_positions'] = 10000 * top['count'] / max(total, 1)
            results[class_name, n] = top
    return results

def compare_lexical_features(eda, stop_words):
    """أحسب الكلمات الوظيفية والترقيم والمصطلحات المحددة مسبقًا."""
    # ملاحظة الطالب: المقارنة المعجمية والأسلوبية
    # ملاحظة: أمثلة محددة مسبقًا لمصطلحات بحثية؛ ليست كلمات مختارة من الاختبار.
    terms = ['بحث','دراسة','نتائج','تحليل','منهج','بيانات']
    function_rows, punctuation_rows, term_rows=[],[],[]
    for c in LABELS:
        subset=eda.loc[eda['class_name'].eq(c)]
        tokens=Counter(w for text in subset['text_normalized'] for w in text.split())
        total=sum(tokens.values())
        function=Counter({w:n for w,n in tokens.items() if w in stop_words})
        for w,n in function.most_common(15):
            function_rows.append({'class_name':c,'word':w,'count':n,'per_10000_words':10000*n/max(total,1)})
        # ملاحظة: مقام الترقيم هو عدد الحروف الأصلية قبل إزالة العلامات.
        chars=sum(map(len,subset['text_raw']))
        for sign in ['،',',','.','؛','؟','!','«','»',':']:
            n=sum(t.count(sign) for t in subset['text_raw'])
            punctuation_rows.append({'class_name':c,'sign':sign,'count':n,'per_1000_chars':1000*n/max(chars,1)})
        for term in terms:
            w=normalize_arabic(term)
            term_rows.append({'class_name':c,'term':term,'count':tokens[w],
                              'per_10000_words':10000*tokens[w]/max(total,1)})
    function_table=pd.DataFrame(function_rows)
    punctuation_table=pd.DataFrame(punctuation_rows)
    term_table=pd.DataFrame(term_rows)
    return function_table, punctuation_table, term_table
