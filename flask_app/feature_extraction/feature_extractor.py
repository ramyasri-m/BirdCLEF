# Required Libraries
import librosa
import numpy as np
from scipy import signal

class FeatureExtractor:
    def __init__(self, delta_width=3, lowcut=1000, highcut=10000, sr=44100, n_mfcc=13, n_fft=1024, frame_length=1024, roll_percent=0.85):
        # Initialize parameters
        self.delta_width = delta_width
        self.lowcut = lowcut
        self.highcut = highcut
        self.sr = sr
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.frame_length = frame_length
        self.roll_percent = roll_percent
        
    def bandpass_filter(self, audio_file_path):
        # Load audio file and apply bandpass filter
        y, sr = librosa.load(audio_file_path, sr=self.sr)
        sos = signal.butter(10, [self.lowcut, self.highcut], btype='band', fs=sr, output='sos')
        filtered = signal.sosfilt(sos, y)
        return filtered, sr
    
    def extract_features(self, audio_file_path):
        # Apply bandpass filter
        y, sr = self.bandpass_filter(audio_file_path)

        features = {}
        
        # MFCC
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=self.n_mfcc, n_fft=self.n_fft)
        for i in range(self.n_mfcc):
            features[f'mfcc_mean_{i+1}'] = np.mean(mfcc[i])
            features[f'mfcc_std_{i+1}'] = np.std(mfcc[i])

        # Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=self.n_fft)
        for i in range(chroma.shape[0]):
            features[f'chroma_mean_{i+1}'] = np.mean(chroma[i])
            features[f'chroma_std_{i+1}'] = np.std(chroma[i])

        # Spectral Contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft)
        for i in range(spectral_contrast.shape[0]):
            features[f'spectral_contrast_mean_{i+1}'] = np.mean(spectral_contrast[i])
            features[f'spectral_contrast_std_{i+1}'] = np.std(spectral_contrast[i])

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft)
        features['spectral_centroid_mean'] = np.mean(spectral_centroid)
        features['spectral_centroid_std'] = np.std(spectral_centroid)

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft)
        features['spectral_bandwidth_mean'] = np.mean(spectral_bandwidth)
        features['spectral_bandwidth_std'] = np.std(spectral_bandwidth)

        # Zero-Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y, frame_length=self.frame_length)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)

        # Delta MFCC
        if mfcc.shape[1] >= self.delta_width:
            delta_mfcc = librosa.feature.delta(mfcc, width=self.delta_width)
            for i in range(delta_mfcc.shape[0]):
                features[f'delta_mfcc_mean_{i+1}'] = np.mean(delta_mfcc[i])
                features[f'delta_mfcc_std_{i+1}'] = np.std(delta_mfcc[i])

        # Spectral Rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, roll_percent=self.roll_percent)
        features['spectral_rolloff_mean'] = np.mean(spectral_rolloff)
        features['spectral_rolloff_std'] = np.std(spectral_rolloff)

        # Spectral Flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft)
        features['spectral_flatness_mean'] = np.mean(spectral_flatness)
        features['spectral_flatness_std'] = np.std(spectral_flatness)

        # RMS Energy
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_std'] = np.std(rms)
        
        return features

    def process_sample(self, audio_file_path, latitude, longitude):
        # Extract audio features
        features = self.extract_features(audio_file_path)

        # Normalize the audio features using manual range normalization
        feature_values = np.array(list(features.values()))
        L, U = feature_values.min(), feature_values.max()  # Determine global min and max
        if U != L:
            normalized_features = (feature_values - L) / (U - L)
        else:
            normalized_features = np.zeros_like(feature_values)  # Avoid divide-by-zero if all values are the same

        try:
            latitude = float(latitude)  # Ensure numerical input
            longitude = float(longitude)  # Ensure numerical input
            # Convert latitude and longitude to radians
            latitude_rad = np.radians(latitude)
            longitude_rad = np.radians(longitude)
            # Continue processing with converted values
        except ValueError:
            raise ValueError("Latitude and longitude must be valid numbers.")
    
        # Transform to polar coordinates
        x = np.cos(latitude_rad) * np.cos(longitude_rad)
        y = np.cos(latitude_rad) * np.sin(longitude_rad)
        z = np.sin(latitude_rad)

        # Combine normalized audio features and polar coordinates
        final_features = np.concatenate([normalized_features, [x, y, z]])

        # Return the processed features or result
        return final_features
