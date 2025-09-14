# Improved MindRove Gesture Classification System

## 🎯 Problem Analysis

You identified the key issue with the original rule-based system:
- **Window averaging dilutes gesture signals** - Quick actions like mining/placing get averaged out over large time windows
- **Low sample counts** within windows reduce signal quality
- **Mining and placing actions are very unstable** due to these limitations

## 🚀 Solution: ML-Based Burst Detection

### Key Improvements

1. **2-Second Burst Analysis** instead of sliding windows
2. **Peak Detection** instead of averaging (captures gesture intensity)
3. **3 Training Samples** per gesture with rest periods
4. **Combined IMU + EMG** features for better discrimination
5. **ML Classification** (SVM, XGBoost, Random Forest)

### System Components

#### 1. Data Collection (`collect_ml_training_data.py`)
```bash
python collect_ml_training_data.py --output training_data.npz --ip 192.168.4.1
```
- Collects 2-second bursts for each gesture
- 3 samples per gesture class
- 3-second rest periods between samples
- Extracts comprehensive features (90+ per sample)

#### 2. ML Training (`train_ml_classifier.py`)
```bash
python train_ml_classifier.py --data training_data.npz --output best_model.pkl
```
- Trains SVM, XGBoost, and Random Forest
- Hyperparameter optimization with cross-validation
- Feature importance analysis
- Saves best performing model

#### 3. Real-time Classification (`ml_key_mapper.py`)
```bash
python ml_key_mapper.py --model best_model.pkl --ip 192.168.4.1
```
- Real-time gesture prediction
- Confidence thresholding
- Proper action cooldowns
- Statistics tracking

#### 4. Rule-Based Improvements (`improved_key_mapper.py`)
- Enhanced burst detection without ML
- Better feature extraction
- More robust thresholds

## 📊 Feature Engineering

### IMU Features (36 total)
- **Statistical**: mean, std, min, max, peak-to-peak, RMS (6 × 6 channels)
- **Cross-channel**: acceleration magnitude stats, gyro magnitude stats
- **Temporal**: motion onset, peak timing, gesture duration

### EMG Features (32 total)
- **Statistical**: mean, std, max, integrated EMG (4 × 8 channels)
- **Activation**: total EMG activation across all channels

### Spectral Features (6 total)
- **Frequency**: dominant frequencies for accel/gyro axes

### Total: 90+ features per 2-second burst

## 🎮 Gesture Mapping

| Gesture | Action | Detection Method |
|---------|--------|------------------|
| `swing_down` | Left Click (Mining) | Peak downward accel + rotation |
| `arm_up` | Right Click (Placing) | Peak upward accel + EMG |
| `reach_forward` | Right Click (Placing) | Forward motion pattern |
| `rotate_left/right` | (Future use) | Gyro rotation patterns |
| `idle` | No action | Low confidence/motion |

## 🔧 Installation & Setup

### 1. Install Additional Dependencies
```bash
pip install -r minecraft_cli/ml_requirements.txt
```

### 2. Connect MindRove
```bash
# Connect to MindRove WiFi hotspot (MindRove_armband_XXXX, password: #mindrove)
# Or use your network IP if device is configured for it
```

### 3. Collect Training Data
```bash
cd minecraft_cli
python collect_ml_training_data.py --ip 192.168.4.1 --output my_training_data.npz
```
Follow the prompts to perform each gesture 3 times with rest periods.

### 4. Train ML Model
```bash
python train_ml_classifier.py --data my_training_data.npz --output my_model.pkl --plot
```

### 5. Use with Minecraft
```bash
python ml_key_mapper.py --model my_model.pkl --ip 192.168.4.1 --confidence 0.7
```

## ⚙️ Configuration Options

### Confidence Threshold
- Default: `0.7` (70% confidence required)
- Lower = more sensitive, more false positives
- Higher = less sensitive, fewer false positives

### Action Cooldown
- Default: `0.5` seconds between actions
- Prevents rapid-fire clicking
- Adjustable in code

### Burst Duration
- Default: `2.0` seconds analysis windows
- Captures full gesture patterns
- Balanced between responsiveness and accuracy

## 🎯 Expected Improvements

### Over Rule-Based System:
1. **Higher Accuracy**: ML models learn complex patterns
2. **Better Stability**: Peak detection vs averaging
3. **EMG Integration**: Muscle activation adds discrimination
4. **Personalization**: Trained on your specific gestures
5. **Confidence Scoring**: Only act on high-confidence predictions

### Performance Targets:
- **Accuracy**: 85-95% (vs 60-70% rule-based)
- **Response Time**: <0.5s after gesture completion
- **False Positives**: <5% with proper confidence threshold

## 🔍 Troubleshooting

### Low Accuracy
- Collect more training data
- Ensure consistent gesture performance
- Check EMG electrode placement
- Try different confidence thresholds

### High Latency
- Reduce burst duration (but may hurt accuracy)
- Check MindRove connection quality
- Verify sampling rate matches expected (500 Hz)

### False Positives
- Increase confidence threshold
- Increase action cooldown period
- Retrain with better neutral/idle samples

## 📈 Advanced Usage

### Custom Gestures
Add new gesture classes by modifying:
1. `GESTURE_CLASSES` in `collect_ml_training_data.py`
2. Action mapping in `ml_key_mapper.py`

### Feature Selection
Use feature importance plots to identify best features and reduce dimensionality.

### Model Tuning
Experiment with different algorithms and hyperparameters in the training script.

---

This system addresses your core issue: **unstable mining/placing due to averaging over large windows**. By using 2-second bursts with peak detection and ML classification, you should see much more stable and accurate gesture recognition! 🚀