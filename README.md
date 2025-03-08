# AI Text Processing with Batch Perplexity & Entropy Computation

This repository contains Python scripts to:
1. Process large datasets in batches,
2. Compute per-token perplexities and entropy scores using two language models,
3. Store the processed results,
4. Match scores back to the original DataFrame.

---

## Overview

**Key Components:**

- **`process_batch`**  
  Processes a list of texts in batches to improve efficiency.

- **`process2scores`**  
  Generates perplexity and cross-entropy metrics for a single text or a batch of texts.  
  - Returns a list of four arrays per text: `[l1, l2, l3, l4]`, representing token-wise scores.

- **`process_dataframe_in_batches`**  
  Uses true batch processing to handle large DataFrames, saving intermediate progress to a JSON file after every 10 batches (or final batch).

- **`match_scores_with_dataframe`**  
  Matches the computed scores back to the original DataFrame rows, either by DataFrame index or by a specific ID column.

---

## Setup & Requirements

make sure to add your {token} in the pre-proccessing file

1. **Python Version**  
   - This code has been tested with Python 3.8+ (older versions may work but are untested).

2. **Dependencies**  
   - `pandas` for DataFrame manipulation  
   - `tqdm` for progress bars  
   - `numpy` for numerical arrays  
   - `json` (part of the standard library)  
   - `os` and `time` (part of the standard library)  
   - `torch` or a similar library if `bino` depends on PyTorch (depending on how `bino` is implemented)  

3. **The `bino` Library**  
   - Ensure you have the `bino` library installed or available in your environment.  
   - `bino.tokenize(texts)`: Tokenizes a list of texts.  
   - `bino.get_logits(tokenized)`: Retrieves logits from the models.  
   - `bino.tokenizer.pad_token_id`: Used for identifying the padding token in the tokenized outputs.

4. **Additional Custom Code**  
   - Functions such as `per_token_perplexity()` and `per_token_entropy()` must be defined or imported before running the main script.

---

## Usage

### 1. Processing a DataFrame in Batches

Suppose you have a `pandas.DataFrame` called `df` with a column named `'text'`. You can process it in batches and save the output to JSON by calling:

```python
process_dataframe_in_batches(
    df,
    output_filename='path/to/output/results',
    batch_size=8,
    start_index=0
)
```

### 2. Matching Scores Back to the DataFrame

After processing, match scores back to the original DataFrame:

```python
import json

with open('path/to/output/results_processed.json', 'r') as f:
    combined_data = json.load(f)

enhanced_df = match_scores_with_dataframe(combined_data, df, use_index=False, df_id_column='_id')
enhanced_df = enhanced_df.dropna(subset=['scores'])

print(len(enhanced_df))
```

---

## Best Practices and Tips

1. **Check for Alignment**  
   Ensure the index or ID used to match scores with DataFrame rows is consistent.

2. **Monitoring Long Jobs**  
   Monitor memory usage and disk space for large datasets.

3. **Resuming Progress**  
   Restart processing from a specific row index if needed.

---

## Contributing

Contributions are welcome! Feel free to open a pull request or file an issue for enhancements or bug fixes.

---

## License

This project is released under the [MIT License](LICENSE). You are free to use, modify, and distribute this code.

