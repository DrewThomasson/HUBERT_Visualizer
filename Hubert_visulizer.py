from datasets import load_dataset
import torch
import torchaudio
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.manifold import TSNE
import pandas as pd
import seaborn as sns
from itertools import groupby
from operator import itemgetter

# Define languages
languages = ['en', 'fr', 'es']
audio_samples = []

# Load data for each language, limiting download to one shard
for lang in languages:
    print(f"Loading data for language: {lang} (one shard only)")
    dataset = load_dataset(
        "mozilla-foundation/common_voice_11_0", 
        lang, 
        split="train", 
        streaming=True  # Use streaming mode
    )
    dataset = dataset.take(50)  # Limit to 50 samples for testing purposes

    for item in dataset:
        audio_samples.append({
            'audio': item['audio']['array'],
            'sampling_rate': item['audio']['sampling_rate'],
            'language': lang
        })

# Load the pre-trained multilingual Wav2Vec2 model
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
model.eval()

embeddings = []
labels = []

# Function to group contiguous frames into syllable intervals
def group_frames(frames, hop_length, sampling_rate):
    syllable_intervals = []
    for k, g in groupby(enumerate(frames), lambda ix : ix[0] - ix[1]):
        group = list(map(itemgetter(1), g))
        start_frame = group[0]
        end_frame = group[-1]
        start_time = start_frame * hop_length / sampling_rate
        end_time = (end_frame * hop_length + hop_length) / sampling_rate
        syllable_intervals.append((start_time, end_time))
    return syllable_intervals

# Process each audio sample
for sample in audio_samples:
    audio = sample['audio']
    sampling_rate = sample['sampling_rate']
    language = sample['language']
    
    # Resample if necessary
    if sampling_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sampling_rate, target_sr=16000)
        sampling_rate = 16000
    
    # Normalize audio
    audio = audio / np.max(np.abs(audio))
    
    # Syllable segmentation (using short-time energy)
    frame_length = int(0.025 * sampling_rate)  # 25 ms
    hop_length = int(0.010 * sampling_rate)    # 10 ms

    energy = np.array([
        sum(abs(audio[i:i+frame_length]**2))
        for i in range(0, len(audio) - frame_length + 1, hop_length)
    ])

    # Normalize energy
    energy = energy / np.max(energy)

    # Thresholding to find syllable boundaries
    energy_threshold = 0.3  # Adjust this threshold based on audio
    frames_above_threshold = np.where(energy > energy_threshold)[0]

    # Group contiguous frames
    syllable_intervals = group_frames(frames_above_threshold, hop_length, sampling_rate)

    # Extract embeddings for each syllable
    for idx, (start_time, end_time) in enumerate(syllable_intervals):
        # Extract syllable audio segment
        start_sample = int(start_time * sampling_rate)
        end_sample = int(end_time * sampling_rate)
        syllable_audio = audio[start_sample:end_sample]
        
        # Skip very short segments
        min_length = 400  # Minimum length for Wav2Vec2 input (25 ms at 16kHz)
        if len(syllable_audio) < min_length:
            padding = min_length - len(syllable_audio)
            syllable_audio = np.pad(syllable_audio, (0, padding), mode='constant')
        
        # Prepare input for Wav2Vec2
        input_values = feature_extractor(syllable_audio, return_tensors="pt", sampling_rate=16000).input_values
        
        # Validate input shape
        if input_values.shape[-1] < 16:  # Ensure sufficient temporal length
            continue
        
        # Get embeddings
        with torch.no_grad():
            outputs = model(input_values)
            hidden_states = outputs.last_hidden_state  # Shape: [1, seq_len, hidden_size]
        
        # Aggregate embeddings (mean over time steps)
        syllable_embedding = torch.mean(hidden_states, dim=1).squeeze().numpy()
        embeddings.append(syllable_embedding)
        labels.append({
            'language': language,
            'sample_id': sample,
            'syllable_idx': idx
        })

# Convert embeddings to numpy array
embeddings_np = np.array(embeddings)

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42, perplexity=5)
embeddings_2d = tsne.fit_transform(embeddings_np)

# Prepare DataFrame for visualization
df = pd.DataFrame(embeddings_2d, columns=['Dim1', 'Dim2'])
df['language'] = [label['language'] for label in labels]

# Visualize the embeddings
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Dim1', y='Dim2', hue='language', palette='deep', s=100)
plt.title("2D Visualization of Syllable Embeddings Using Wav2Vec2 Multilingual Model", fontsize=16)
plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.legend(title='Language')
plt.grid(True)
plt.show()
