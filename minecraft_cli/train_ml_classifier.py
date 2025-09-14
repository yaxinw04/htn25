#!/usr/bin/env python3
"""
ML Classifier Training for MindRove Gestures
============================================

Trains SVM and XGBoost classifiers on collected gesture data.
Compares performance and saves the best model.

Usage: python train_ml_classifier.py --data training_data.csv --output best_model.pkl
"""

import argparse
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, Tuple
import pandas as pd

class MindRoveMLTrainer:
    """Trains and evaluates ML classifiers for MindRove gesture recognition"""
    
    def __init__(self):
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = []
        self.gesture_labels = []
        
    def load_data(self, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess time-series training data with windowing"""
        print(f"📥 Loading and preprocessing time-series data from {data_file}")
        
        if data_file.endswith('.csv'):
            # Load raw time-series data
            import pandas as pd
            df = pd.read_csv(data_file)
            
            # Preprocess with sliding windows
            X, y = self._preprocess_timeseries(df, data_file)
            
            return X, y
        else:
            # Load from NPZ (legacy format - not supported for time-series)
            raise ValueError("NPZ format not supported for time-series data. Use CSV format.")
    
    def _preprocess_timeseries(self, df: pd.DataFrame, data_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess time-series data with sliding windows and statistical features"""
        print("🔄 Preprocessing time-series data with sliding windows...")
        
        # Parameters - reduced overlap to prevent overfitting
        window_size = 1.0  # 0.5 second windows
        stride = 0.4       # 0.4 second stride (less overlap)
        sample_rate = 100  # Approximate sample rate from data collection
        
        # Convert to sample counts
        window_samples = int(window_size * sample_rate)  # ~50 samples
        stride_samples = int(stride * sample_rate)       # ~20 samples
        
        # Sensor columns (exclude metadata)
        sensor_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                      'emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7']
        
        windowed_features = []
        windowed_labels = []
        
        # Group by gesture instances (gesture_name + chronological order)
        gesture_instances = []
        for gesture_name in df['gesture_name'].unique():
            gesture_df = df[df['gesture_name'] == gesture_name].copy()
            gesture_df = gesture_df.sort_values('timestamp')
            
            # Group by gesture occurrences (when timestamps reset)
            instance_groups = []
            current_group = []
            last_timestamp = -1
            
            for _, row in gesture_df.iterrows():
                if row['timestamp'] < last_timestamp:  # New instance
                    if current_group:
                        instance_groups.append(current_group)
                    current_group = []
                current_group.append(row)
                last_timestamp = row['timestamp']
            
            if current_group:
                instance_groups.append(current_group)
            
            gesture_instances.extend(instance_groups)
        
        print(f"📊 Found {len(gesture_instances)} gesture instances")
        
        # Process each gesture instance with sliding windows
        for instance in gesture_instances:
            if len(instance) < window_samples:
                continue  # Skip instances too short for windowing
            
            instance_df = pd.DataFrame(instance)
            gesture_id = instance_df['gesture_id'].iloc[0]
            
            # Extract sensor data
            sensor_data = instance_df[sensor_cols].values  # Shape: (time_points, sensors)
            
            # Create sliding windows
            for start_idx in range(0, len(sensor_data) - window_samples + 1, stride_samples):
                end_idx = start_idx + window_samples
                window_data = sensor_data[start_idx:end_idx]  # Shape: (window_samples, sensors)
                
                # Extract statistical features from window
                features = self._extract_window_features(window_data)
                
                windowed_features.append(features)
                windowed_labels.append(gesture_id)
        
        X = np.array(windowed_features)
        y = np.array(windowed_labels)
        
        print(f"✅ Windowing complete:")
        print(f"   Original time points: {len(df)}")
        print(f"   Windowed samples: {X.shape[0]}")
        print(f"   Features per window: {X.shape[1]}")
        
        # Print class distribution
        unique, counts = np.unique(y, return_counts=True)
        gesture_names = {0: "idle", 1: "swing_down", 2: "arm_up", 3: "wrist_flex_left", 4: "wrist_supinate_right"}
        print(f"📈 Windowed class distribution:")
        for class_id, count in zip(unique, counts):
            class_name = gesture_names.get(class_id, f"class_{class_id}")
            print(f"   {class_name}: {count} windows ({count/len(y)*100:.1f}%)")
        
        # Save preprocessed data to new CSV (don't overwrite original)
        preprocessed_file = data_file.replace('.csv', '_preprocessed.csv')  # Fixed variable name
        self._save_preprocessed_data(X, y, preprocessed_file)
        
        return X, y
    
    def _extract_window_features(self, window_data: np.ndarray) -> np.ndarray:
        """Extract statistical features from a time window"""
        features = []
        
        # For each sensor channel
        for ch in range(window_data.shape[1]):
            channel_data = window_data[:, ch]
            
            # Statistical features
            features.append(np.mean(channel_data))      # Mean
            features.append(np.std(channel_data))       # Standard deviation  
            features.append(np.min(channel_data))       # Minimum
            features.append(np.max(channel_data))       # Maximum
            features.append(np.max(channel_data) - np.min(channel_data))  # Range
            features.append(np.sqrt(np.mean(channel_data**2)))  # RMS
        
        # Cross-channel features
        accel_data = window_data[:, :3]  # Accelerometer
        gyro_data = window_data[:, 3:6]  # Gyroscope
        emg_data = window_data[:, 6:]    # EMG
        
        # Magnitude features
        accel_mag = np.sqrt(np.sum(accel_data**2, axis=1))
        gyro_mag = np.sqrt(np.sum(gyro_data**2, axis=1))
        
        features.extend([
            np.mean(accel_mag), np.std(accel_mag), np.max(accel_mag),
            np.mean(gyro_mag), np.std(gyro_mag), np.max(gyro_mag),
            np.mean(np.sum(emg_data, axis=1))  # Total EMG activation
        ])
        
        return np.array(features)
    
    def _save_preprocessed_data(self, X: np.ndarray, y: np.ndarray, output_file: str):
        """Save preprocessed windowed features to CSV"""
        print(f"💾 Saving preprocessed features to {output_file}")
        
        # Create feature names
        sensor_names = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z',
                       'emg_0', 'emg_1', 'emg_2', 'emg_3', 'emg_4', 'emg_5', 'emg_6', 'emg_7']
        stat_names = ['mean', 'std', 'min', 'max', 'range', 'rms']
        
        feature_names = []
        for sensor in sensor_names:
            for stat in stat_names:
                feature_names.append(f"{sensor}_{stat}")
        
        # Add cross-channel features
        feature_names.extend([
            'accel_mag_mean', 'accel_mag_std', 'accel_mag_max',
            'gyro_mag_mean', 'gyro_mag_std', 'gyro_mag_max',
            'emg_total_mean'
        ])
        
        # Create DataFrame
        df_processed = pd.DataFrame(X, columns=feature_names)
        df_processed['gesture_id'] = y
        
        gesture_names = {0: "idle", 1: "swing_down", 2: "arm_up", 3: "wrist_flex_left", 4: "wrist_supinate_right"}
        df_processed['gesture_name'] = [gesture_names.get(label, f"unknown_{label}") for label in y]
        
        # Save to CSV
        df_processed.to_csv(output_file, index=False)
        self.feature_names = feature_names
        
        print(f"✅ Preprocessed data saved with {len(feature_names)} features per sample")
        
        return X, y
        
    def prepare_data(self, X: np.ndarray, y: np.ndarray, test_size: float = 0.2) -> Tuple:
        """Prepare data for training with reduced data leakage"""
        print(f"🔄 Preparing data for training...")
        
        # Handle any NaN or infinite values
        X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Simple stratified split - the instance grouping was too complex
        # With reduced window overlap (0.4s stride vs 0.2s), correlation is reduced
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        print(f"📊 Train/Test Split:")
        print(f"   Training: {X_train.shape[0]} windows")
        print(f"   Testing: {X_test.shape[0]} windows")
        print(f"   Features: {X_train.shape[1]}")
        
        # Check class distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        unique_test, counts_test = np.unique(y_test, return_counts=True)
        
        print(f"📈 Class distribution:")
        gesture_names = {0: "idle", 1: "swing_down", 2: "arm_up", 3: "wrist_flex_left", 4: "wrist_supinate_right"}
        print(f"   Train: {dict(zip([gesture_names.get(u, f'class_{u}') for u in unique_train], counts_train))}")
        print(f"   Test:  {dict(zip([gesture_names.get(u, f'class_{u}') for u in unique_test], counts_test))}")
        
        return X_train, X_test, y_train, y_test
    
    def train_svm_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Train SVM classifier with hyperparameter tuning"""
        print(f"🔧 Training SVM classifier...")
        
        # Create pipeline with scaling
        svm_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('svm', SVC(random_state=42, probability=True))
        ])
        
        # Hyperparameter grid for SVM
        param_grid = {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'svm__kernel': ['rbf', 'poly', 'linear']
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            svm_pipeline, param_grid, cv=5, scoring='accuracy', 
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best SVM parameters: {grid_search.best_params_}")
        print(f"📊 Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def train_xgboost_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Train XGBoost classifier with hyperparameter tuning"""
        print(f"🚀 Training XGBoost classifier...")
        
        # Create pipeline (XGBoost handles scaling internally but we'll include it for consistency)
        xgb_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('xgb', xgb.XGBClassifier(random_state=42, eval_metric='mlogloss'))
        ])
        
        # Hyperparameter grid for XGBoost
        param_grid = {
            'xgb__n_estimators': [100, 200, 300],
            'xgb__max_depth': [3, 5, 7],
            'xgb__learning_rate': [0.01, 0.1, 0.2],
            'xgb__subsample': [0.8, 0.9, 1.0],
            'xgb__colsample_bytree': [0.8, 0.9, 1.0]
        }
        
        # Grid search with cross-validation
        grid_search = GridSearchCV(
            xgb_pipeline, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best XGBoost parameters: {grid_search.best_params_}")
        print(f"📊 Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
        
    def train_random_forest_classifier(self, X_train: np.ndarray, y_train: np.ndarray) -> Pipeline:
        """Train Random Forest classifier as baseline"""
        print(f"🌲 Training Random Forest classifier...")
        
        rf_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('rf', RandomForestClassifier(random_state=42, n_jobs=-1))
        ])
        
        param_grid = {
            'rf__n_estimators': [100, 200, 300],
            'rf__max_depth': [None, 5, 10, 15],
            'rf__min_samples_split': [2, 5, 10],
            'rf__min_samples_leaf': [1, 2, 4]
        }
        
        grid_search = GridSearchCV(
            rf_pipeline, param_grid, cv=5, scoring='accuracy',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"✅ Best RF parameters: {grid_search.best_params_}")
        print(f"📊 Best CV score: {grid_search.best_score_:.3f}")
        
        return grid_search.best_estimator_
    
    def evaluate_model(self, model: Pipeline, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
        """Evaluate a trained model"""
        print(f"📈 Evaluating {model_name}...")
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 {model_name} Accuracy: {accuracy:.3f}")
        
        # Classification report
        gesture_names = {0: "idle", 1: "swing_down", 2: "arm_up", 3: "reach_forward", 
                        4: "rotate_left", 5: "rotate_right"}
        target_names = [gesture_names.get(i, f"class_{i}") for i in sorted(np.unique(y_test))]
        
        print(f"📊 {model_name} Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        return {
            'model': model,
            'accuracy': accuracy,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'y_test': y_test
        }
    
    def plot_confusion_matrices(self, results: Dict[str, Dict], output_dir: str = "."):
        """Plot confusion matrices for all models"""
        n_models = len(results)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 5))
        
        if n_models == 1:
            axes = [axes]
        
        gesture_names = {0: "idle", 1: "swing_down", 2: "arm_up", 3: "reach_forward", 
                        4: "rotate_left", 5: "rotate_right"}
        
        for idx, (model_name, result) in enumerate(results.items()):
            y_test = result['y_test']
            y_pred = result['y_pred']
            
            # Create confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            
            # Plot
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=[gesture_names.get(i, f"C{i}") for i in sorted(np.unique(y_test))],
                       yticklabels=[gesture_names.get(i, f"C{i}") for i in sorted(np.unique(y_test))])
            axes[idx].set_title(f'{model_name}\nAccuracy: {result["accuracy"]:.3f}')
            axes[idx].set_xlabel('Predicted')
            axes[idx].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"💾 Confusion matrices saved to {output_dir}/confusion_matrices.png")
    
    def analyze_feature_importance(self, model: Pipeline, model_name: str, top_k: int = 20):
        """Analyze feature importance for tree-based models"""
        try:
            if hasattr(model.named_steps, 'rf'):  # Random Forest
                importances = model.named_steps['rf'].feature_importances_
            elif hasattr(model.named_steps, 'xgb'):  # XGBoost
                importances = model.named_steps['xgb'].feature_importances_
            else:
                print(f"⚠️  Feature importance not available for {model_name}")
                return
            
            # Get top features
            indices = np.argsort(importances)[::-1][:top_k]
            
            print(f"🏆 Top {top_k} features for {model_name}:")
            for i, idx in enumerate(indices):
                print(f"   {i+1:2d}. {self.feature_names[idx]:30s} ({importances[idx]:.4f})")
                
            # Plot feature importance
            plt.figure(figsize=(10, 6))
            plt.title(f'Feature Importance - {model_name}')
            plt.bar(range(top_k), importances[indices])
            plt.xticks(range(top_k), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(f"feature_importance_{model_name.lower().replace(' ', '_')}.png", 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error analyzing feature importance: {e}")
    
    def save_best_model(self, model: Pipeline, model_name: str, accuracy: float, output_file: str):
        """Save the best performing model"""
        model_info = {
            'model': model,
            'model_name': model_name,
            'accuracy': accuracy,
            'feature_names': self.feature_names,
            'gesture_names': {0: "idle", 1: "swing_down", 2: "arm_up", 3: "reach_forward", 
                            4: "rotate_left", 5: "rotate_right"},
            'training_info': {
                'scaler_mean': model.named_steps['scaler'].mean_ if 'scaler' in model.named_steps else None,
                'scaler_scale': model.named_steps['scaler'].scale_ if 'scaler' in model.named_steps else None
            }
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(model_info, f)
            
        print(f"💾 Best model ({model_name}) saved to {output_file}")
        print(f"🎯 Final accuracy: {accuracy:.3f}")
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict]:
        """Train and evaluate all models"""
        results = {}
        
        # Train SVM
        svm_model = self.train_svm_classifier(X_train, y_train)
        results['SVM'] = self.evaluate_model(svm_model, X_test, y_test, 'SVM')
        
        # Train XGBoost
        xgb_model = self.train_xgboost_classifier(X_train, y_train)
        results['XGBoost'] = self.evaluate_model(xgb_model, X_test, y_test, 'XGBoost')
        
        # Train Random Forest (baseline)
        rf_model = self.train_random_forest_classifier(X_train, y_train)
        results['Random Forest'] = self.evaluate_model(rf_model, X_test, y_test, 'Random Forest')
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        print(f"\n🏆 Best Model: {best_model_name} (Accuracy: {best_accuracy:.3f})")
        
        self.best_model = results[best_model_name]['model']
        self.best_score = best_accuracy
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Train ML classifiers for MindRove gestures")
    parser.add_argument("--data", required=True, help="Input training data file (.csv or .npz)")
    parser.add_argument("--output", default="best_model.pkl", help="Output model file (.pkl)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set size (default: 0.2)")
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    print("🤖 MindRove ML Classifier Training")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MindRoveMLTrainer()
    
    # Load data
    X, y = trainer.load_data(args.data)
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y, args.test_size)
    
    # Train all models
    print(f"\n🚀 Training multiple classifiers...")
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Model comparison
    print(f"\n📊 Model Comparison:")
    print(f"{'Model':<15} {'Accuracy':<10} {'Status'}")
    print("-" * 35)
    for model_name, result in results.items():
        status = "⭐ BEST" if result['accuracy'] == trainer.best_score else ""
        print(f"{model_name:<15} {result['accuracy']:<10.3f} {status}")
    
    # Feature importance analysis
    print(f"\n🔍 Feature Importance Analysis:")
    for model_name, result in results.items():
        if model_name in ['XGBoost', 'Random Forest']:
            trainer.analyze_feature_importance(result['model'], model_name)
    
    # Generate plots if requested
    if args.plot:
        trainer.plot_confusion_matrices(results)
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    trainer.save_best_model(
        results[best_model_name]['model'], 
        best_model_name,
        results[best_model_name]['accuracy'],
        args.output
    )
    
    print(f"\n🎉 Training complete!")
    print(f"💡 Next steps:")
    print(f"   1. Test the model: python test_ml_classifier.py --model {args.output}")
    print(f"   2. Use in Minecraft: python ml_key_mapper.py --model {args.output}")

if __name__ == '__main__':
    main()