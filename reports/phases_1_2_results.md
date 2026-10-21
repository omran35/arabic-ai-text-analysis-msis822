# Phase 1 and Phase 2 - Computed findings
Dataset: KFUPM-JRCAI/arabic-generated-abstracts; revision: 584e43a346914a4cb1a51f9679f903284b65c36e.
Source rows: 8,388; raw text slots: 41,940; unique usable texts: 36,478.
Invalid slots: 0; no-Arabic rows: 4; contradictory-label rows: 117.
Exact duplicate occurrences removed: 5,341.
EDA training texts: 25,577. Validation and test were reserved before linguistic exploration.
## Descriptive class differences
- word_count: Human = 118.3900; AI = 100.9678; AI is lower on average.
- avg_word_length: Human = 5.0744; AI = 5.3333; AI is higher on average.
- avg_sentence_length: Human = 54.8423; AI = 21.1921; AI is lower on average.
- ttr: Human = 0.7926; AI = 0.7673; AI is lower on average.
- ttr_first_50: Human = 0.8829; AI = 0.8564; AI is lower on average.
- function_words_per_100_words: Human = 24.1054; AI = 21.1880; AI is lower on average.
- punctuation_per_100_chars: Human = 1.7413; AI = 1.6526; AI is lower on average.
## Interpretation and limitations
- Class sizes are unequal; frequency plots use within-class normalized rates.
- Raw text is retained; punctuation and sentence statistics precede destructive cleaning.
- TTR depends on length; first-50-token TTR is reported with eligible counts.
- Repeated human abstracts are deduplicated; all related variants share a source group.
- Normalization collisions stay in one group; ISRI may cause further collisions that must be audited before future modeling.
- Grouping covers shared originals and normalized duplicates, not undetected semantic near-duplicates.
- Punctuation-based sentence segmentation and whitespace tokenization are approximate.
- ISRI is stemming, not lemmatization; stopwords approximate function words.
- No significance tests or classifier-performance claims are made.
- This corpus contains research abstracts; domain and generation-method confounding remain.
- No models, feature-selection procedures or final test evaluation were run.