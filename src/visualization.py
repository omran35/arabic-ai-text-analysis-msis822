"""عرض نتائج التحليل وحفظ الرسوم مع دعم العربية."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from wordcloud import WordCloud

from .utils import LABELS, COLORS, SEED


def configure_plots():
    """أطبق إعدادات الرسم عند استدعائها فقط."""
    plt.rcParams.update({'figure.dpi': 110, 'axes.spines.top': False,
                         'axes.spines.right': False, 'font.size': 10})


def ar(text):
    """أضبط تشكيل الحروف واتجاهها للعرض دون تغيير النص المخزن."""
    import arabic_reshaper
    from bidi.algorithm import get_display
    return get_display(arabic_reshaper.reshape(str(text)))


def save_figure(figure, path, *, dpi=170):
    """أحفظ الشكل وأعيد مساره."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches='tight')
    return path


def finish(fig, name, figures_dir):
    """أحفظ الرسم ثم أعرضه في مخرجات خلية الاستدعاء."""
    save_figure(fig, Path(figures_dir) / f'{name}.png')
    plt.show()
    plt.close(fig)

def plot_class_distribution(class_counts, figures_dir):
    """أعرض أعداد الفئتين قبل ضبط الجودة وبعده."""
    fig, axes = plt.subplots(1,2,figsize=(10,3.5))
    for ax, col, title in zip(axes,['raw_text_slots','unique_exact_texts'],
                             ['Before quality cleaning','After quality cleaning and deduplication']):
        counts = class_counts[col]
        ax.bar(counts.index,counts.values,color=[COLORS[x] for x in LABELS])
        ax.set(title=title,ylabel='Number of texts')
        ax.bar_label(ax.containers[0],padding=3)
        ax.margins(y=0.16)
    finish(fig,'phase1_class_distribution', figures_dir)

def plot_linguistic_comparison(eda, figures_dir):
    """أعرض المقاييس الستة وتوزيع طول نصوص التدريب."""
    # ملاحظة الطالب: قراءة الرسوم الإحصائية
    # ملاحظة: كل مربع يقارن مقياسًا واحدًا بين الفئتين.
    plot_metrics = [('word_count','Words per abstract'),('avg_word_length','Average word length'),
                    ('avg_sentence_length','Words per sentence'),('ttr','Type-token ratio'),
                    ('ttr_first_50','TTR: first 50 words'),
                    ('punctuation_per_100_chars','Punctuation per 100 raw characters')]
    fig,axes = plt.subplots(2,3,figsize=(13,7))
    for ax,(metric,title) in zip(axes.flat,plot_metrics):
        samples = [eda.loc[eda['class_name'].eq(c),metric].dropna() for c in LABELS]
        boxes = ax.boxplot(samples,patch_artist=True,showfliers=False)
        ax.set_xticks([1,2],LABELS)
        for patch,c in zip(boxes['boxes'],LABELS):
            patch.set_facecolor(COLORS[c]); patch.set_alpha(.75)
        ax.set_title(title)
    fig.suptitle('Training texts only | Box plots omit outlier markers, not observations',y=1.02)
    finish(fig,'phase2_linguistic_comparison', figures_dir)
    fig,ax=plt.subplots(figsize=(8,4))
    for c in LABELS:
        # ملاحظة: الكثافة النسبية تمنع أن يبدو التوزيع أكبر لمجرد أن عدد النصوص أكثر.
        ax.hist(eda.loc[eda['class_name'].eq(c),'word_count'],bins=45,density=True,
                alpha=.5,label=c,color=COLORS[c])
    ax.set(xlabel='Words per abstract',ylabel='Density',title='Length distributions (training only)')
    ax.legend(); finish(fig,'phase2_word_count_distribution', figures_dir)

def plot_ngrams(results, figures_dir):
    """أرسم أكثر الكلمات والعبارات شيوعًا من جداول التحليل."""
    for n in (1, 2, 3):
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
        for ax, class_name in zip(axes, LABELS):
            shown = results[class_name, n].iloc[::-1]
            ax.barh([ar(s) for s in shown['ngram']], shown['per_10000_positions'], color=COLORS[class_name])
            ax.set(title=f'{class_name}: top {n}-grams', xlabel='Occurrences per 10,000 positions')
        finish(fig, f'phase2_top_{n}grams', figures_dir)

def plot_wordclouds(eda, figures_dir, *, seed=SEED):
    """أعرض سحابة لكل فئة بكلمات غير مجذرة واتجاه عربي."""
    Path(figures_dir).mkdir(parents=True, exist_ok=True)
    # ملاحظة الطالب: سحابات الكلمات
    # ملاحظة: أبحث عن خط يدعم العربية في Colab أو Windows.
    font_candidates=[
        Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'),
        Path('/usr/share/fonts/truetype/noto/NotoNaskhArabic-Regular.ttf'),
        Path('C:/Windows/Fonts/arial.ttf'),
        Path(font_manager.findfont('DejaVu Sans'))]
    FONT = next((p for p in font_candidates if p.exists()),None)
    assert FONT is not None,'An Arabic-compatible TrueType font is required.'
    fig,axes=plt.subplots(1,2,figsize=(13,4.5))
    for ax,c in zip(axes,LABELS):
        counts=Counter(w for text in eda.loc[eda['class_name'].eq(c),'text_no_stopwords'] for w in text.split())
        assert counts,f'No tokens available for {c}'
        # Merge any display-equivalent tokens rather than overwriting their frequencies.
        # ملاحظة: تشكيل الحروف واتجاه العرض يخص الرسم فقط، ولا يغير النص المخزن.
        display_counts=Counter()
        for word,count in counts.items():
            display_counts[ar(word)]+=count
        cloud=WordCloud(width=1100,height=650,background_color='white',font_path=str(FONT),
                        collocations=False,max_words=90,random_state=seed).generate_from_frequencies(display_counts)
        ax.imshow(cloud,interpolation='bilinear'); ax.axis('off'); ax.set_title(c)
    finish(fig,'phase2_wordcloud_comparison', figures_dir)

def plot_lexical_comparison(punctuation_table, term_table, figures_dir):
    """أقارن معدلات الترقيم والمصطلحات البحثية بين الفئتين."""
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,frame,key,value,title in [
        (axes[0],punctuation_table,'sign','per_1000_chars','Punctuation per 1,000 raw characters'),
        (axes[1],term_table,'term','per_10000_words','Research terms per 10,000 words')]:
        # ملاحظة: أرتب الجدول إلى عمود لكل فئة لتسهيل الرسم والمقارنة.
        pivot=frame.pivot(index=key,columns='class_name',values=value).reindex(columns=LABELS)
        pivot.index=[ar(x) for x in pivot.index]
        pivot.plot.bar(ax=ax,color=[COLORS[c] for c in LABELS],rot=0)
        ax.set(title=title,xlabel='')
    finish(fig,'phase2_lexical_comparison', figures_dir)
