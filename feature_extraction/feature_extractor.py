# Import required libraries
import librosa
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from joblib import Parallel, delayed
from collections import Counter
import re
import scipy.signal as signal
import os
from tqdm import tqdm
import gc  # Garbage collector

class FeatureExtractor:

# Load an audio file (replace 'audio_file.wav' with the path to your audio file)
audio_path = '/kaggle/input/birdclef-2022/train_audio/afrsil1/XC125458.ogg'
y, sr = librosa.load(audio_path)

# Create a dictionary to store the extracted features
features = {}

# 1. Extract MFCC (Mel-Frequency Cepstral Coefficients)
# MFCCs are used to capture the timbral texture of an audio signal.
# 'n_mfcc' is the number of MFCC features to extract; 13 is commonly used.
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
features['MFCCs'] = mfcc
print("MFCCs Shape:", mfcc.shape)

# 2. Chroma Feature
# Chroma features are used to capture pitch class (similar to musical notes).
# They help recognize harmonic structures in audio.
chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
features['Chroma'] = chroma_stft
print("Chroma Shape:", chroma_stft.shape)

# 3. Spectral Centroid
# This measures the "brightness" of a sound by calculating the center of mass of the spectrum.
# Higher values indicate a "brighter" sound.
spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
features['Spectral Centroid'] = spectral_centroid
print("Spectral Centroid Shape:", spectral_centroid.shape)

# 4. Spectral Bandwidth
# Spectral Bandwidth measures the range of frequencies in the sound.
# Higher values indicate a sound with a wider frequency range.
spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
features['Spectral Bandwidth'] = spectral_bandwidth
print("Spectral Bandwidth Shape:", spectral_bandwidth.shape)

# 5. Spectral Contrast
# This measures the difference in amplitude between peaks and valleys in the spectrum.
# It can help distinguish between sounds with different timbres.
spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
features['Spectral Contrast'] = spectral_contrast
print("Spectral Contrast Shape:", spectral_contrast.shape)

# 6. Zero Crossing Rate (ZCR)
# ZCR calculates how often the audio signal crosses the zero amplitude level.
# Noisy sounds have high ZCR, while smoother sounds have low ZCR.
zcr = librosa.feature.zero_crossing_rate(y)
features['Zero Crossing Rate'] = zcr
print("Zero Crossing Rate Shape:", zcr.shape)

# 7. Root Mean Square Energy (RMSE)
# RMSE measures the power (or loudness) of the signal, indicating sound intensity over time.
rmse = librosa.feature.rms(y=y)
features['RMSE'] = rmse
print("RMSE Shape:", rmse.shape)

# 8. Mel Spectrogram
# A Mel spectrogram represents the energy of different frequency bands over time on the Mel scale.
# This is used in many audio tasks as it captures frequency and time-based patterns.
mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
features['Mel Spectrogram'] = mel_spectrogram
print("Mel Spectrogram Shape:", mel_spectrogram.shape)

# 9. Harmonic-to-Noise Ratio (HNR)
# HNR measures the ratio of harmonic content to noise content in the audio signal.
# It's useful for distinguishing between tonal and non-tonal sounds.
harmonic, percussive = librosa.effects.hpss(y)  # Separates harmonic and percussive elements
hnr = librosa.effects.harmonic(y) / librosa.effects.percussive(y)
features['Harmonic-to-Noise Ratio'] = hnr
print("Harmonic-to-Noise Ratio Shape:", hnr.shape)

# Print a summary of all extracted features
for feature_name, feature_value in features.items():
    print(f"{feature_name} extracted with shape: {feature_value.shape}")

# Display only the first set of coefficients for inspection, e.g., MFCC
print("\nSample MFCC Coefficients:\n", mfcc[:, :5])  # Print the first 5 columns of MFCCs

# Load the metadata and taxonomy data
metadata_path = '/kaggle/input/birdclef-2022/train_metadata.csv'
taxonomy_path = '/kaggle/input/birdclef-2022/eBird_Taxonomy_v2021.csv'

# Read the CSV files into DataFrames
metadata_df = pd.read_csv(metadata_path)
taxonomy_df = pd.read_csv(taxonomy_path)

# Check if all values in primary_label are in SPECIES_CODE
metadata_labels = set(metadata_df['primary_label'].unique())
taxonomy_codes = set(taxonomy_df['SPECIES_CODE'].unique())

# Check if both sets are equal
if metadata_labels == taxonomy_codes:
    print("All values in 'primary_label' match with 'SPECIES_CODE'.")
else:
    print("Values in 'primary_label' do not match with 'SPECIES_CODE'.")
    print("Mismatch found:")
    print(f"Primary labels not in taxonomy: {metadata_labels - taxonomy_codes}")
    #print(f"Taxonomy codes not in metadata: {taxonomy_codes - metadata_labels}")

# Load the metadata
# Load the metadata and taxonomy files
metadata_path = '/kaggle/input/birdclef-2022/train_metadata.csv'
taxonomy_path = '/kaggle/input/birdclef-2022/eBird_Taxonomy_v2021.csv'
audio_base_path = '/kaggle/input/birdclef-2022/train_audio/'

metadata_df = pd.read_csv(metadata_path)
taxonomy_df = pd.read_csv(taxonomy_path)

# Filter the DataFrame to include only samples with rating >= 3
filtered_metadata_df = metadata_df[metadata_df['rating'] >= 3]

# Display the total number of filtered samples
filtered_samples = len(filtered_metadata_df)
print(f"Total number of samples with rating >= 3: {filtered_samples}")

# List of columns to analyze
columns_to_analyze = ['secondary_labels', 'type']

# Function to analyze each column with individual string components, case-insensitive and space-insensitive
def analyze_column_individual_case_insensitive(df, column_name):
    print(f"\nAnalyzing individual values within the column (case-insensitive and space-insensitive): {column_name}")
    
    # Initialize a Counter to count each unique item across all entries
    item_counter = Counter()
    
    # Iterate over each entry in the column
    for entry in df[column_name].dropna():
        # Convert the string to a list, make each item lowercase, and remove spaces
        items = [re.sub(r'\s+', '', item.lower()) for item in eval(entry)]  # Lowercase and remove spaces
        # Update the counter with items in the list
        item_counter.update(items)
    
    # Print total unique items and counts for each
    #unique_items = len(item_counter)
    #print(f"Total number of unique items: {unique_items}")
    #print(f"\nCount of occurrences for each unique item (case-insensitive and space-insensitive):\n{item_counter}")
    
    return item_counter

# Analyze the 'type' column
type_counts = analyze_column_individual_case_insensitive(filtered_metadata_df, 'type')

# Define noisy types to consider
noisy_terms = ['wing', 'wings', 'water', 'splash', 'rain', 'ground', 'background', 'noise', 'anthropogenic', 'traffic', 'street']

# Filter types that contain any of the noisy terms
noisy_types = [item for item in type_counts if any(term in item for term in noisy_terms)]

print("\nTypes identified as containing noise (including specified terms):")
print(noisy_types)

# Function to filter DataFrame by removing rows where all 'type' values match noisy types
def filter_noisy_rows(df, noisy_types):
    def has_only_noisy_types(type_list):
        # Clean and normalize type_list
        items = [re.sub(r'\s+', '', item.lower()) for item in eval(type_list)]
        # Check if all items are in noisy_types
        return all(item in noisy_types for item in items)
    
    # Filter the DataFrame
    filtered_df = df[~df['type'].apply(has_only_noisy_types)]
    
    return filtered_df

# Apply the filter to remove rows with only noisy types
filtered_metadata_df_cleaned = filter_noisy_rows(filtered_metadata_df, noisy_types)

# Display the total number of filtered samples
filtered_samples = len(filtered_metadata_df_cleaned)
print(f"\nTotal number of samples without noisy types: {filtered_samples}")

# Analyze the 'type' column again for unique items after filtering
#unique_items_after_filtering = analyze_column_individual_case_insensitive(filtered_metadata_df_cleaned, 'type')

# Display the filtered DataFrame without noisy types
#print("\nFiltered DataFrame without noisy types:")
#print(filtered_metadata_df_cleaned)

# Merge with taxonomy on scientific name
merged_df = pd.merge(filtered_metadata_df_cleaned, taxonomy_df, left_on='scientific_name', right_on='SCI_NAME', how='left')

# Drop specified columns after merging
merged_df = merged_df.drop(columns=['primary_label', 'PRIMARY_COM_NAME', 'secondary_labels', 'author', 'license', 'rating', 'REPORT_AS', 'SCI_NAME', 'time', 'url', 'SPECIES_GROUP'])

# Display DataFrames after filtering and merging.
print("DataFrame after filtering and merging:")
print(merged_df.head())

# List of columns to analyze
columns_to_analyze = ['CATEGORY', 'ORDER1', 'FAMILY', 'SPECIES_CODE', 'common_name', 'scientific_name', 'TAXON_ORDER']

# Function to analyze each column
def analyze_column(df, column_name):
    print(f"\nColumn: {column_name}")
    
    # Total number of unique values
    unique_values = df[column_name].nunique()
    print(f"Total number of unique values: {unique_values}")
    
    # Number of missing values
    missing_values = df[column_name].isnull().sum()
    print(f"Number of missing values: {missing_values}")
    
    # Count of occurrences for each unique value
    value_counts = df[column_name].value_counts()
    print(f"\nCount of occurrences for each unique value:\n{value_counts}")

# Analyze each column in the list
for column in columns_to_analyze:
    analyze_column(merged_df, column)

# Encode hierarchical taxonomy levels (e.g., Order, Family)
label_encoder = LabelEncoder()
#merged_df['Category'] = label_encoder.fit_transform(merged_df['CATEGORY'])
merged_df['Order'] = label_encoder.fit_transform(merged_df['ORDER1'])
merged_df['Family'] = label_encoder.fit_transform(merged_df['FAMILY'])
#merged_df['Species_code'] = label_encoder.fit_transform(merged_df['SPECIES_CODE'])
#merged_df['Pri_com_name'] = label_encoder.fit_transform(merged_df['PRIMARY_COM_NAME'])
#merged_df['Scientific_name'] = label_encoder.fit_transform(merged_df['SCI_NAME'])

# Save the final DataFrame to a CSV file
merged_df.to_csv("merged_data.csv", index=False)
    
# Randomly sample 1,000 entries from the filtered DataFrame
sampled_merged_df = merged_df.sample(n=1000, random_state=42)

# Display DataFrames after encoding.
print("DataFrame after encoding:")
print(merged_df.head())

# List of columns to analyze
columns_to_analyze = ['Order', 'Family']

# Function to analyze each column
def analyze_column(df, column_name):
    print(f"\nColumn: {column_name}")
    
    # Total number of unique values
    unique_values = df[column_name].nunique()
    print(f"Total number of unique values: {unique_values}")
    
    # Number of missing values
    missing_values = df[column_name].isnull().sum()
    print(f"Number of missing values: {missing_values}")
    
    # Count of occurrences for each unique value
    value_counts = df[column_name].value_counts()
    print(f"\nCount of occurrences for each unique value:\n{value_counts}")

# Analyze each column in the list
for column in columns_to_analyze:
    analyze_column(merged_df, column)

# Define the base path for audio files
base_path = '/kaggle/input/birdclef-2022/train_audio/'

delta_width = 3
# Bandpass filter function for typical birdsong frequencies (1-10 kHz)
def bandpass_filter(y, sr, lowcut=1000, highcut=10000):
    sos = signal.butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
    filtered = signal.sosfilt(sos, y)
    return filtered

# Function to extract audio features
def extract_features(y, sr):
    features = {}
    
    # Apply bandpass filter
    y = bandpass_filter(y, sr)
    
    # MFCC - Mean and Standard Deviation across 13 coefficients
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024)
    for i in range(13):
        features[f'mfcc_mean_{i+1}'] = np.mean(mfcc[i])
        features[f'mfcc_std_{i+1}'] = np.std(mfcc[i])

    # Chroma - Mean and Standard Deviation across 12 chroma features
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024)
    for i in range(12):
        features[f'chroma_mean_{i+1}'] = np.mean(chroma[i])
        features[f'chroma_std_{i+1}'] = np.std(chroma[i])
    
    # Spectral Contrast - Mean and Standard Deviation across bands
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024)
    for i in range(spectral_contrast.shape[0]):
        features[f'spectral_contrast_mean_{i+1}'] = np.mean(spectral_contrast[i])
        features[f'spectral_contrast_std_{i+1}'] = np.std(spectral_contrast[i])

    # Spectral Centroid - Mean and Standard Deviation
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024)
    features['spectral_centroid_mean'] = np.mean(spectral_centroid)
    features['spectral_centroid_std'] = np.std(spectral_centroid)
    
    # Spectral Bandwidth - Mean and Standard Deviation
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024)
    features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
    features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)
    
    # Zero-Crossing Rate - Mean and Standard Deviation
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024)
    features['zcr_mean'] = np.mean(zcr)
    features['zcr_std'] = np.std(zcr)
    
    # Additional Features
    # Delta MFCCs - Mean and Standard Deviation
    if mfcc.shape[1] >= delta_width:  # Check if MFCC frames are enough for delta calculation
        delta_mfcc = librosa.feature.delta(mfcc, width=delta_width)
        for i in range(delta_mfcc.shape[0]):
            features[f'delta_mfcc_mean_{i+1}'] = np.mean(delta_mfcc[i])
            features[f'delta_mfcc_std_{i+1}'] = np.std(delta_mfcc[i])
    else:
        print(f"Skipping delta calculation for audio with insufficient frames (width={delta_width})")
    
    # Spectral Rolloff - Mean and Standard Deviation
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, roll_percent=0.85)
    features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
    features['spectral_rolloff_std'] = np.std(spectral_rolloff)
    
    # Spectral Flatness - Mean and Standard Deviation
    spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=1024)
    features['spectral_flatness_mean'] = np.mean(spectral_flatness)
    features['spectral_flatness_std'] = np.std(spectral_flatness)
    
    # Root Mean Square (RMS) Energy - Mean and Standard Deviation
    rms = librosa.feature.rms(y=y)
    features['rms_mean'] = np.mean(rms)
    features['rms_std'] = np.std(rms)
    
    return features

# Function to process data in batches
def process_in_batches(df, batch_size=100):
    all_features = []
    num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
    
    for batch_num in tqdm(range(num_batches), desc="Processing Batches"):
        batch_df = df.iloc[batch_num * batch_size : (batch_num + 1) * batch_size]
        
        batch_features = []
        for filename in batch_df['filename']:
            file_path = os.path.join(base_path, filename)
            
            y, sr = librosa.load(file_path, sr=44100)  # Set sample rate to 44100 Hz
            features = extract_features(y, sr)
            batch_features.append(features)
        
        # Convert batch features to DataFrame
        features_df = pd.DataFrame(batch_features)
        
        # Concatenate features with the original batch DataFrame
        batch_combined = pd.concat([batch_df.reset_index(drop=True), features_df], axis=1)
        all_features.append(batch_combined)
        
        # Clear memory
        del batch_features, y
        gc.collect()
    
    # Concatenate all batches to form the final DataFrame
    final_data = pd.concat(all_features, ignore_index=True)
    return final_data

# Apply batch processing to the DataFrame
extra_data = process_in_batches(merged_df, batch_size=100)

# Save the final DataFrame to a CSV file
extra_data.to_csv("extra_data.csv", index=False)

# Display DataFrames after feature extraction
print("DataFrame after feature extraction:")
print(extra_data.head())
print(final_data.head())

# Load the dataset
file_path = '/kaggle/input/birdclef/extra_data.csv'
df = pd.read_csv(file_path)

# Convert latitude and longitude to radians
df['latitude_rad'] = np.radians(df['latitude'])
df['longitude_rad'] = np.radians(df['longitude'])

# Transform to polar coordinates
df['x'] = np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
df['y'] = np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
df['z'] = np.sin(df['latitude_rad'])

# Drop the original latitude and longitude columns
df = df.drop(columns=['latitude', 'longitude', 'latitude_rad', 'longitude_rad'])

audio_feature_columns = [col for col in df.columns if col.startswith('mfcc') or 
                   col.startswith('chroma') or col.startswith('spectral') or 
                   col.startswith('zcr') or col.startswith('delta') or 
                   col.startswith('rms')]

# Normalize the audio feature columns to the range [0, 1]
audio_scaler = MinMaxScaler()
df[audio_feature_columns] = audio_scaler.fit_transform(df[audio_feature_columns])

# Drop Unecessary columns
df = df.drop(columns=['type', 'scientific_name', 'common_name', 'filename', 'TAXON_ORDER', 'CATEGORY', 'SPECIES_CODE', 'ORDER1', 'FAMILY'])

# Save the final preprocessed DataFrame
preprocessed_file_path = '/kaggle/working/extra_preprocessed_data.csv'
df.to_csv(preprocessed_file_path, index=False)

# Display a sample of the DataFrame
print("Sample of the preprocessed DataFrame:")
print(df.head())

print(f"\nPreprocessed data saved to {preprocessed_file_path}")
