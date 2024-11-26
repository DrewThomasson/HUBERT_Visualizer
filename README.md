# HUBERT_Visualizer
test script for visualizing dataset with HUBERT

## Syllable Embeddings Visualization

This script processes multilingual audio samples to extract syllable-level embeddings using a pre-trained Wav2Vec2 model. It visualizes these embeddings in a 2D space using t-SNE.

# Sample output
![sample_output](https://github.com/user-attachments/assets/0b85b6d8-4e82-42af-b418-91beae4ed074)



### Features
- Loads and processes audio samples from the Common Voice dataset.
- Segments audio into syllables using short-time energy.
- Extracts embeddings for each syllable using the Wav2Vec2 model.
- Reduces dimensions of embeddings with t-SNE for visualization.
- Visualizes embeddings by language in a scatter plot.

### Requirements
Install dependencies with:
```bash
pip install -r requirements.txt
```

### Usage
Run the script to:
1. Download multilingual audio samples (English, French, Spanish).
2. Preprocess audio and extract syllable embeddings.
3. Generate a scatter plot of embeddings colored by language.

### Notes
- Adjust `energy_threshold` and `perplexity` parameters for optimal results with your dataset.
