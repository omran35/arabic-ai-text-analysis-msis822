"""Regression checks for preprocessing, source grouping and phrase boundaries."""
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from src.analysis import linguistic_stats, ngram_counts
from src.data_preparation import (
    check_column_consistency, combine_datasets, load_raw_datasets,
    prepare_labeled_texts, deduplicate_corpus, reserve_partitions,
)
from src.data_preprocessing import normalize_arabic, preprocess_text, prepare_training_data
from src.utils import METHODS, TEXT_COLUMNS


class PipelineTests(unittest.TestCase):
    def test_arabic_boundaries_and_original_text(self):
        self.assertEqual(normalize_arabic('أَحْمَد إِلَى الْمَدْرَسَةِ'), 'احمد الي المدرسة')
        self.assertEqual(normalize_arabic('علم،بحث AI 123'), 'علم بحث')
        self.assertEqual(normalize_arabic('ة ء ؤ ئ'), 'ة ء ؤ ئ')
        self.assertEqual(normalize_arabic(None), '')
        text = 'إِنَّ هَذِهِ دِرَاسَةٌ عَرَبِيَّةٌ.'
        result = preprocess_text(text, {'ان', 'هذه'})
        self.assertEqual(result['text_no_stopwords'], 'دراسة عربية')
        self.assertEqual(normalize_arabic(result['text_normalized']), result['text_normalized'])

    def test_csv_loading_and_column_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[2]) as directory:
            for method in METHODS:
                pd.DataFrame([{c: 'نص عربي' for c in TEXT_COLUMNS}]).to_csv(
                    Path(directory) / f'{method}.csv', index=False)
            parts = load_raw_datasets(directory)
            self.assertTrue(check_column_consistency(*parts))
            self.assertEqual(len(combine_datasets(*parts)), 3)
            self.assertNotIn('source_row', parts[0])
            with self.assertRaises(ValueError):
                check_column_consistency(parts[0], parts[1], parts[2].drop(columns=TEXT_COLUMNS[-1]))

    def test_quality_isolation_provenance_and_no_leakage(self):
        # Use Arabic suffixes because normalization removes Latin IDs and digits.
        rows = []
        for suffix in ['الف', 'باء', 'جيم', 'دال', 'هاء', 'واو', 'زاي', 'حاء', 'طاء', 'ياء', 'كاف', 'لام']:
            rows.append({column: f'ملخص {suffix} {word}' for column, word in zip(
                TEXT_COLUMNS, ['بشري', 'اول', 'ثاني', 'ثالث', 'رابع'])})
        frame = pd.DataFrame(rows)
        parts = [frame.copy() for _ in METHODS]
        parts[0].loc[0, TEXT_COLUMNS[1]] = None
        parts[0].loc[1, TEXT_COLUMNS[1]] = 'English only 123'
        parts[0].loc[2, TEXT_COLUMNS[1]] = frame.loc[3, 'original_abstract']
        wide = combine_datasets(*parts)
        long, valid, invalid, no_arabic, conflicts = prepare_labeled_texts(wide)
        self.assertEqual(len(invalid), 1)
        self.assertEqual(len(no_arabic), 1)
        self.assertEqual(set(conflicts['label']), {0, 1})
        self.assertTrue(set(valid['text_normalized']).isdisjoint(conflicts['text_normalized']))
        corpus, counts = deduplicate_corpus(valid, long)
        self.assertTrue(corpus['text_id'].is_unique)
        self.assertLess(len(corpus), len(valid))
        self.assertTrue(corpus['occurrences'].gt(1).any())
        self.assertEqual(int(counts['unique_exact_texts'].sum()), len(corpus))
        reserved, _ = reserve_partitions(corpus)
        again, _ = reserve_partitions(corpus)
        self.assertTrue(reserved['partition'].equals(again['partition']))
        self.assertEqual(reserved.groupby('group_id')['partition'].nunique().max(), 1)
        self.assertEqual(reserved.groupby('text_normalized')['partition'].nunique().max(), 1)
        eda = prepare_training_data(reserved, {'ملخص'})
        self.assertTrue(eda['partition'].eq('train').all())
        self.assertTrue(set(eda['text_id']).isdisjoint(reserved.loc[reserved['partition'].ne('train'), 'text_id']))
        self.assertNotIn('partition', corpus)

    def test_ngrams_do_not_cross_texts_or_deleted_stopwords(self):
        counts, opportunities = ngram_counts(['علم في بحث', 'نص عربي'], 2, stop_words={'في'})
        self.assertEqual(opportunities, 3)
        self.assertNotIn('علم بحث', counts)
        self.assertNotIn('بحث نص', counts)
        self.assertEqual(counts['علم في'], 1)
        counts, _ = ngram_counts(['في من'], 2, stop_words={'في', 'من'})
        self.assertFalse(counts)

    def test_statistics_use_raw_punctuation_and_missing_short_ttr(self):
        row = {'text_raw': 'علم علم. بحث!', 'text_normalized': 'علم علم بحث', 'text_no_stopwords': 'علم علم بحث'}
        result = linguistic_stats(row, {'بحث'})
        self.assertEqual(result['word_count'], 3)
        self.assertEqual(result['sentence_count'], 2)
        self.assertAlmostEqual(result['ttr'], 2 / 3)
        self.assertTrue(pd.isna(result['ttr_first_50']))
        self.assertGreater(result['punctuation_per_100_chars'], 0)


if __name__ == '__main__':
    unittest.main()
