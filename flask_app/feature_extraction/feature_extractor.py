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
    
    # Load the metadata and taxonomy files
    metadata_path = 'C:/Users/nivet/Documents/birdclef-2022/train_metadata.csv'
    taxonomy_path = 'C:/Users/nivet/Documents/birdclef-2022/eBird_Taxonomy_v2021.csv'
    audio_base_path = 'C:/Users/nivet/Documents/birdclef-2022/train_audio'
    
    metadata_df = pd.read_csv(metadata_path)
    taxonomy_df = pd.read_csv(taxonomy_path)
    
    # Filter the DataFrame to include only samples with rating >= 3
    filtered_metadata_df = metadata_df[metadata_df['rating'] >= 3]
        
    # Analyze the 'type' column
    type_counts = analyze_column_individual_case_insensitive(filtered_metadata_df, 'type')
    
    # Define noisy types to consider
    noisy_terms = ['wing', 'wings', 'water', 'splash', 'rain', 'ground', 'background', 'noise', 'anthropogenic', 'traffic', 'street']
    
    # Filter types that contain any of the noisy terms
    noisy_types = [item for item in type_counts if any(term in item for term in noisy_terms)]
    
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
    
    # Merge with taxonomy on scientific name
    merged_df = pd.merge(filtered_metadata_df_cleaned, taxonomy_df, left_on='scientific_name', right_on='SCI_NAME', how='left')
    
    # Drop specified columns after merging
    merged_df = merged_df.drop(columns=['primary_label', 'PRIMARY_COM_NAME', 'secondary_labels', 'author', 'license', 'rating', 'REPORT_AS', 'SCI_NAME', 'time', 'url', 'SPECIES_GROUP'])
    
    # Encode hierarchical taxonomy levels (e.g., Order, Family)
    label_encoder = LabelEncoder()
    #merged_df['Category'] = label_encoder.fit_transform(merged_df['CATEGORY'])
    merged_df['Order'] = label_encoder.fit_transform(merged_df['ORDER1'])
    merged_df['Family'] = label_encoder.fit_transform(merged_df['FAMILY'])
    #merged_df['Species_code'] = label_encoder.fit_transform(merged_df['SPECIES_CODE'])
    #merged_df['Pri_com_name'] = label_encoder.fit_transform(merged_df['PRIMARY_COM_NAME'])
    #merged_df['Scientific_name'] = label_encoder.fit_transform(merged_df['SCI_NAME'])
    
    # Save the final DataFrame to a CSV file
    merged_df.to_csv("C:/Users/nivet/Documents/birdclef-2022/merged_data.csv", index=False)
    
    # Define the base path for audio files
    base_path = 'C:/Users/nivet/Documents/birdclef-2022/train_audio'
    
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
    extra_data.to_csv("C:/Users/nivet/Documents/birdclef-2022/extra_data.csv", index=False)
    
    # Load the dataset
    file_path = 'C:/Users/nivet/Documents/birdclef-2022/extra_data.csv'
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
    preprocessed_file_path = 'C:/Users/nivet/Documents/birdclef-2022/extra_preprocessed_data.csv'
    df.to_csv(preprocessed_file_path, index=False)
