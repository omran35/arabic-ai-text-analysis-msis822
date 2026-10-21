"""إعدادات المصدر والمجلدات المشتركة؛ لا ينفذ هذا الملف تدريبًا أو تقييمًا."""

from pathlib import Path
from importlib.metadata import version
import json
import shutil
import sys

DATASET_ID = 'KFUPM-JRCAI/arabic-generated-abstracts'
REVISION = '584e43a346914a4cb1a51f9679f903284b65c36e'
SEED = 822
METHODS = ('by_polishing', 'from_title', 'from_title_and_content')
TEXT_COLUMNS = (
    'original_abstract', 'allam_generated_abstract', 'jais_generated_abstract',
    'llama_generated_abstract', 'openai_generated_abstract',
)
PROJECT_DIRECTORIES = (
    'data/raw', 'data/processed', 'data/external', 'notebooks', 'src', 'models',
    'reports/figures', 'reports/tables', 'reports/presentations', 'docs',
)


def create_project_directories(root: str | Path) -> Path:
    """أجهز مجلدات المشروع وأعيد مساره، دون إنشاء بيانات أو نماذج وهمية."""
    root = Path(root)
    for folder in PROJECT_DIRECTORIES:
        (root / folder).mkdir(parents=True, exist_ok=True)
    return root

LABELS = ['Human', 'AI']
COLORS = {'Human': '#197887', 'AI': '#c26a3c'}

def save_table(frame, name, tables_dir, index=True):
    """أحفظ جدول CSV بترميز مناسب للعربية في Excel."""
    path = Path(tables_dir) / f'{name}.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=index, encoding='utf-8-sig')
    return path

def build_findings_report(eda, metrics, audit):
    """أبني التفسير من المتوسطات والأعداد المحسوبة فعليًا."""
    means=eda.groupby('class_name')[metrics].mean().reindex(LABELS)
    observations=[]
    for metric in ['word_count','avg_word_length','avg_sentence_length','ttr','ttr_first_50',
                   'function_words_per_100_words','punctuation_per_100_chars']:
        h,a=means.loc['Human',metric],means.loc['AI',metric]
        direction='higher' if a>h else ('lower' if a<h else 'equal')
        observations.append(f'- {metric}: Human = {h:.4f}; AI = {a:.4f}; AI is {direction} on average.')
    report='\n'.join([
        '# Phase 1 and Phase 2 - Computed findings',
        f'Dataset: {DATASET_ID}; revision: {REVISION}.',
        f'Source rows: {audit["source_rows"]:,}; raw text slots: {audit["raw_text_slots"]:,}; unique usable texts: {audit["unique_texts"]:,}.',
        f'Invalid slots: {audit["invalid_slots"]:,}; no-Arabic rows: {audit["no_arabic_rows"]:,}; contradictory-label rows: {audit["conflicting_rows"]:,}.',
        f'Exact duplicate occurrences removed: {audit["exact_duplicate_rows_removed"]:,}.',
        f'EDA training texts: {len(eda):,}. Validation and test were reserved before linguistic exploration.',
        '## Descriptive class differences',*observations,
        '## Interpretation and limitations',
        '- Class sizes are unequal; frequency plots use within-class normalized rates.',
        '- Raw text is retained; punctuation and sentence statistics precede destructive cleaning.',
        '- TTR depends on length; first-50-token TTR is reported with eligible counts.',
        '- Repeated human abstracts are deduplicated; all related variants share a source group.',
        '- Normalization collisions stay in one group; ISRI may cause further collisions that must be audited before future modeling.',
        '- Grouping covers shared originals and normalized duplicates, not undetected semantic near-duplicates.',
        '- Punctuation-based sentence segmentation and whitespace tokenization are approximate.',
        '- ISRI is stemming, not lemmatization; stopwords approximate function words.',
        '- No significance tests or classifier-performance claims are made.',
        '- This corpus contains research abstracts; domain and generation-method confounding remain.',
        '- No models, feature-selection procedures or final test evaluation were run.'
    ])
    return report

def save_run_outputs(project, eda, corpus, audit, metrics):
    """أحفظ التقرير والبيانات وخريطة التقسيم وإصدارات بيئة التشغيل."""
    project = Path(project)
    report = build_findings_report(eda, metrics, audit)
    (project/'reports/phase1_phase2_findings.md').write_text(report,encoding='utf-8')
    # ملاحظة: أحفظ بيانات التدريب المعالجة منفصلة عن الأصل وخريطة التقسيم.
    eda.to_parquet(project/'data/processed/phase2_training.parquet',index=False)
    eda.to_csv(project/'data/processed/phase2_training.csv',index=False,encoding='utf-8-sig')
    corpus[['text_id','group_id','partition','label','class_name','generation_methods','generators']].to_csv(
        project/'data/processed/partition_manifest.csv',index=False,encoding='utf-8-sig')
    # ملاحظة: هذه إصدارات البيئة وقت التشغيل، وتفيد في تكرار العمل لاحقًا.
    packages=['datasets','nltk','pandas','numpy','matplotlib','wordcloud','arabic-reshaper','python-bidi','pyarrow']
    environment={p:version(p) for p in packages}
    (project/'requirements-tested.txt').write_text('\n'.join(f'{p}=={v}' for p,v in environment.items())+'\n',encoding='utf-8')
    (project/'docs/run_metadata.json').write_text(json.dumps(
        {'python':sys.version,'seed':SEED,'dataset_revision':REVISION,'packages':environment},indent=2),encoding='utf-8')
    return report

def package_results(project):
    """أجمع التقارير والبيانات الوصفية في ZIP دون البيانات الكبيرة."""
    project = Path(project)
    bundle=project/'submission_results'
    bundle.mkdir(exist_ok=True)
    for folder in ['reports','docs']:
        shutil.copytree(project/folder,bundle/folder,dirs_exist_ok=True)
    shutil.copy2(project/'requirements-tested.txt',bundle/'requirements-tested.txt')
    archive=shutil.make_archive(str(project/'MSIS822_Phases_1_2_results'),'zip',bundle)
    return archive
