"""إعدادات المصدر والمجلدات المشتركة؛ لا ينفذ هذا الملف تدريبًا أو تقييمًا."""

from pathlib import Path

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
