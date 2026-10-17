"""أدوات عرض عربية وحفظ رسوم المرحلة الثانية دون أي نمذجة."""

from pathlib import Path


def ar(text: object) -> str:
    """أضبط تشكيل الحروف واتجاهها للرسم، دون تغيير النص المخزن."""
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(str(text)))


def save_figure(figure, path: str | Path, *, dpi=170) -> Path:
    """أحفظ الشكل في مسار واضح؛ يعرضه الدفتر عند الحاجة بشكل مستقل."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    return path
