# AI Text Processing with Batch Perplexity & Entropy Computation

This repository contains Python scripts to:
1. Process large datasets in batches.
2. Compute per-token perplexities and entropy scores using two language models.
3. Store the processed results.
4. Match scores back to the original DataFrame for analysis.

---

## Overview

**Key Components:**

- **`process_batch`**: Efficient batch processing of text samples.
- **`process2scores`**: Computes per-token perplexity and cross-entropy scores.
- **`process_dataframe_in_batches`**: Handles large DataFrames efficiently, saving progress incrementally.
- **`match_scores_with_dataframe`**: Matches computed scores back to their original DataFrame entries.

---

## Setup & Requirements

Make sure to add your `{token}` in the pre-processing file.

### Dependencies

- Python 3.8+
- `pandas`, `tqdm`, `numpy`, `json`, `os`, `time`
- `torch` (if `bino` relies on PyTorch)
- `bino` library for text tokenization and logit extraction.

### Additional Code Requirements

Ensure functions such as `per_token_perplexity()` and `per_token_entropy()` are properly defined.

---

![Model Output](https://github.com/user-attachments/assets/be8ab214-4ba3-4844-a843-4bf142fcfbff)

## Usage

### 1. Processing a DataFrame in Batches

```python
process_dataframe_in_batches(
    df,
    output_filename='path/to/output/results',
    batch_size=8,
    start_index=0
)
```

### 2. Matching Scores Back to the DataFrame

```python
import json

with open('path/to/output/results_processed.json', 'r') as f:
    combined_data = json.load(f)

enhanced_df = match_scores_with_dataframe(combined_data, df, use_index=False, df_id_column='_id')
enhanced_df = enhanced_df.dropna(subset=['scores'])

print(len(enhanced_df))
```

---

## Results

Our AI detection model achieved:

- **Training & Validation Accuracy:** 85–90%
- **Test Accuracy:** 75%
- **ROC AUC Score:** ~0.82

These results indicate that our model effectively distinguishes AI-generated text from human-written text using contrastive perplexity analysis and RNN-based classification.

---

## Best Practices & Tips

1. **Ensure Alignment**: Keep consistent indexing for score matching.
2. **Monitor Resource Usage**: Large datasets may require careful memory management.
3. **Resume Processing**: Use `start_index` to avoid redundant computations if interrupted.

---

## Contributing

Contributions are welcome! Open a pull request or file an issue for improvements.

---

## License

This project is released under the [MIT License](LICENSE).

![Keynote Template](https://github.com/user-attachments/assets/7d697047-4687-4a7e-93db-837d8132d799)
