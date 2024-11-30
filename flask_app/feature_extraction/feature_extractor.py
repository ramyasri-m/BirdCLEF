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
    def __init__(self, metadata_path, taxonomy_path, audio_base_path, delta_width=3):
        # Initialize paths
        self.metadata_path = metadata_path
        self.taxonomy_path = taxonomy_path
        self.audio_base_path = audio_base_path
        self.delta_width = delta_width

        # Load metadata and taxonomy DataFrames
        self.metadata_df = pd.read_csv(metadata_path)
        self.taxonomy_df = pd.read_csv(taxonomy_path)

        # Filter metadata for rating >= 3
        self.filtered_metadata_df = self.metadata_df[self.metadata_df['rating'] >= 3]

    def merge_and_process_taxonomy(self):
        merged_df = pd.merge( self.filtered_metadata_df, self.taxonomy_df, left_on='scientific_name', right_on='SCI_NAME', how='left')
        merged_df = merged_df.drop(columns=[ 'primary_label', 'PRIMARY_COM_NAME', 'secondary_labels', 'author', 'license', 'rating', 'REPORT_AS', 'SCI_NAME', 'time', 'url', 'SPECIES_GROUP'])
        
        label_encoder = LabelEncoder()
        merged_df['Order'] = label_encoder.fit_transform(merged_df['ORDER1'])
        merged_df['Family'] = label_encoder.fit_transform(merged_df['FAMILY'])
        merged_df.to_csv("C:/Users/nivet/Documents/birdclef-2022/merged_data.csv", index=False)
        
        return merged_df

    def bandpass_filter(self, y, sr, lowcut=1000, highcut=10000):
        sos = signal.butter(10, [lowcut, highcut], btype='band', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, y)
        
        return filtered

    def extract_features(self, y, sr):
        features = {}
        # Apply bandpass filter
        y = self.bandpass_filter(y, sr)

        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=1024)
        for i in range(13):
            features[f'mfcc_mean_{i+1}'] = np.mean(mfcc[i])
            features[f'mfcc_std_{i+1}'] = np.std(mfcc[i])

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=1024)
        for i in range(12):
            features[f'chroma_mean_{i+1}'] = np.mean(chroma[i])
            features[f'chroma_std_{i+1}'] = np.std(chroma[i])

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=1024)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_mean_{i+1}'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_std_{i+1}'] = np.std(spectral_contrast[i])

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=1024)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=1024)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=1024)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Delta MFCC
        if mfcc.shape[1] >= self.delta_width:
            delta_mfcc = librosa.feature.delta(mfcc, width=self.delta_width)
            for i in range(delta_mfcc.shape[0]):
                features[f'delta_mfcc_mean_{i+1}'] = np.mean(delta_mfcc[i])
                features[f'delta_mfcc_std_{i+1}'] = np.std(delta_mfcc[i])
        else:
            print(f"Skipping Delta MFCC calculation (insufficient frames: {self.delta_width})")

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=1024, roll_percent=0.85)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        # Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=1024)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features

    # Function to process data in batches
    def process_in_batches(self, df, batch_size=100):
        all_features = []
        num_batches = len(df) // batch_size + (1 if len(df) % batch_size != 0 else 0)
        
        for batch_num in tqdm(range(num_batches), desc="Processing Batches"):
            batch_df = df.iloc[batch_num * batch_size : (batch_num + 1) * batch_size]
            
            batch_features = []
            for filename in batch_df['filename']:
                file_path = os.path.join(self.audio_base_path, filename)  # Using self.audio_base_path
                
                y, sr = librosa.load(file_path, sr=44100)  # Set sample rate to 44100 Hz
                features = self.extract_features(y, sr)  # Using self to call extract_features
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

    def process_and_save_data(self, batch_size=100):
        
        # Apply batch processing to the DataFrame
        extra_data = self.process_in_batches(self.filtered_metadata_df, batch_size=batch_size)

        # Save the batch-processed DataFrame to CSV
        extra_data_file_path = "C:/Users/nivet/Documents/birdclef-2022/extra_data.csv"
        extra_data.to_csv(extra_data_file_path, index=False)
        
        # Load the saved dataset
        df = pd.read_csv(extra_data_file_path)
        
        # Convert latitude and longitude to radians
        df['latitude_rad'] = np.radians(df['latitude'])
        df['longitude_rad'] = np.radians(df['longitude'])
        
        # Transform to polar coordinates
        df['x'] = np.cos(df['latitude_rad']) * np.cos(df['longitude_rad'])
        df['y'] = np.cos(df['latitude_rad']) * np.sin(df['longitude_rad'])
        df['z'] = np.sin(df['latitude_rad'])
        
        # Drop the original latitude and longitude columns
        df = df.drop(columns=['latitude', 'longitude', 'latitude_rad', 'longitude_rad'])
        
        # Select only the audio feature columns for normalization
        audio_feature_columns = [col for col in df.columns if col.startswith('mfcc') or 
                                 col.startswith('chroma') or col.startswith('spectral') or 
                                 col.startswith('zcr') or col.startswith('delta') or 
                                 col.startswith('rms')]
        
        # Normalize the audio feature columns to the range [0, 1]
        audio_scaler = MinMaxScaler()
        df[audio_feature_columns] = audio_scaler.fit_transform(df[audio_feature_columns])
        
        # Drop unnecessary columns
        df = df.drop(columns=['type', 'scientific_name', 'common_name', 'filename', 'TAXON_ORDER', 'CATEGORY', 'SPECIES_CODE', 'ORDER1', 'FAMILY'])
        
        # Save the final preprocessed DataFrame to a CSV file
        preprocessed_file_path = 'C:/Users/nivet/Documents/birdclef-2022/extra_preprocessed_data.csv'
        df.to_csv(preprocessed_file_path, index=False)
        
        return df
        
# Define paths for the metadata, taxonomy, and audio base
metadata_path = 'C:/Users/nivet/Documents/birdclef-2022/train_metadata.csv'
taxonomy_path = 'C:/Users/nivet/Documents/birdclef-2022/eBird_Taxonomy_v2021.csv'
audio_base_path = 'C:/Users/nivet/Documents/birdclef-2022/train_audio'

# Create an instance of the FeatureExtractor class
feature_extractor = FeatureExtractor(metadata_path, taxonomy_path, audio_base_path)

# Call the method to process and save the data
preprocessed_data = feature_extractor.process_and_save_data(batch_size=100)

# Optionally, you can inspect the preprocessed data
print(preprocessed_data.head())
