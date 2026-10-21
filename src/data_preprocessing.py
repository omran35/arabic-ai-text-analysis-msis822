"""تطبيع العربية وكلمات الوقف وتجذير ISRI؛ لا تنزيل عند الاستيراد."""

from functools import lru_cache
from pathlib import Path
import re
import unicodedata

DIACRITICS = re.compile(r'[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]')
ARABIC_LETTERS = re.compile(r'[\u0621-\u063a\u0641-\u064a]+')


def canonical(text: str) -> str:
    """أوحد Unicode والمسافات لمقارنة النص الكامل، دون حذف أسلوبه الأصلي."""
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', text)).strip()


def normalize_arabic(text: object) -> str:
    """أزيل التشكيل والتطويل وأوحد الألف والمقصورة وأفصل الحروف غير العربية."""
    if not isinstance(text, str):
        return ''
    text = unicodedata.normalize('NFKC', text)
    text = DIACRITICS.sub('', text).replace('ـ', '')
    text = re.sub('[أإآٱ]', 'ا', text).replace('ى', 'ي')
    return ' '.join(ARABIC_LETTERS.findall(text))


def normalized_stopwords(words) -> set[str]:
    """أطبق تطبيع النص نفسه على قائمة كلمات الوقف."""
    return {token for word in words for token in normalize_arabic(word).split()}


def get_arabic_stopwords(data_dir: str | Path | None = None, *, download=False) -> set[str]:
    """أقرأ قائمة NLTK العربية. التنزيل اختياري وصريح عند عدم توفر المورد."""
    import nltk
    from nltk.corpus import stopwords

    if data_dir is not None:
        data_dir = Path(data_dir)
        if download:
            data_dir.mkdir(parents=True, exist_ok=True)
        if str(data_dir) not in nltk.data.path:
            nltk.data.path.insert(0, str(data_dir))
    if download:
        if not nltk.download('stopwords', download_dir=str(data_dir) if data_dir else None, quiet=True):
            raise RuntimeError('تعذر تنزيل قائمة كلمات الوقف العربية.')
    return normalized_stopwords(stopwords.words('arabic'))


def remove_stopwords(normalized_text: str, stop_words: set[str]) -> str:
    """أحذف كلمات الوقف من نسخة مطبعة، وأبقي الأصل محفوظًا لدى المستدعي."""
    return ' '.join(word for word in normalized_text.split() if word not in stop_words)


@lru_cache(maxsize=1)
def _stemmer():
    from nltk.stem.isri import ISRIStemmer
    return ISRIStemmer()


@lru_cache(maxsize=100000)
def stem_token(token: str) -> str:
    """أعيد جذر ISRI التقريبي للكلمة وأخزن النتيجة لتجنب تكرار الحساب."""
    return _stemmer().stem(token)


def stem_text(text_without_stopwords: str) -> str:
    return ' '.join(stem_token(word) for word in text_without_stopwords.split())


def preprocess_text(text: object, stop_words: set[str]) -> dict[str, str]:
    """أنتج النسخ الثلاث بنفس تسلسل الدفتر: تطبيع ثم وقف ثم تجذير."""
    normalized = normalize_arabic(text)
    without_stopwords = remove_stopwords(normalized, stop_words)
    return {
        'text_normalized': normalized,
        'text_no_stopwords': without_stopwords,
        'text_stemmed': stem_text(without_stopwords),
    }

def prepare_training_data(corpus, stop_words):
    """أعالج قسم التدريب وحده مع إبقاء النص الأصلي والتقسيم."""
    eda = corpus.loc[corpus['partition'].eq('train')].copy().reset_index(drop=True)
    eda['text_no_stopwords'] = eda['text_normalized'].map(lambda text: remove_stopwords(text, stop_words))
    eda['text_stemmed'] = eda['text_no_stopwords'].map(stem_text)
    assert not eda['text_stemmed'].str.contains(r'[A-Za-z0-9]', regex=True).any()
    return eda
