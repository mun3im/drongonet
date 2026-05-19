# XiaoChirp Architecture Sweep Analysis Report

## Executive Summary

Analysis of 35 experimental configurations across 4 architectures reveals that **7_low_power** architecture with GlobalAveragePooling achieves the best accuracy (94.72%) with the smallest model size (5.12 KB). This analysis provides insights for designing higher-accuracy architectures.

---

## Results Summary by Architecture

### 1. 7_low_power (WINNER)
| n_fft | n_mels | Accuracy   | Size (KB) | Infer (ms) | AUC        |
| ----- | ------ | ---------- | --------- | ---------- | ---------- |
| 1024  | 16     | 0.9274     | 5.12      | 0.14       | 0.9787     |
| 1024  | 32     | 0.9372     | 5.12      | 0.26       | 0.9802     |
| 1024  | 48     | 0.9214     | 5.12      | 0.43       | 0.9753     |
| 1024  | 64     | 0.9178     | 5.12      | 0.33       | 0.9705     |
| 1024  | **80** | **0.9472** | 5.12      | 0.72       | **0.9870** |
|       |        |            |           |            |            |

**Key insight**: GlobalAveragePooling keeps model size constant (5.12 KB) regardless of n_mels!

### 2. 1_baseline
| n_fft | n_mels | Accuracy | Size (KB) | Infer (ms) | AUC |
|-------|--------|----------|-----------|------------|------|
| 512 | 16 | 0.9178 | 7.28 | 0.06 | 0.9721 |
| 512 | 32 | 0.9302 | 12.78 | 0.14 | 0.9763 |
| 512 | 48 | 0.9322 | 18.28 | 0.21 | 0.9785 |
| 512 | 80 | 0.9388 | 29.28 | 0.33 | 0.9816 |
| 1024 | 16 | 0.9168 | 7.28 | 0.07 | 0.9700 |
| 1024 | 32 | 0.9316 | 12.78 | 0.13 | 0.9788 |
| 1024 | 48 | 0.9308 | 18.28 | 0.20 | 0.9801 |
| 1024 | 64 | 0.9262 | 23.78 | 0.26 | 0.9777 |
| 1024 | 80 | 0.9370 | 29.28 | 0.34 | 0.9809 |

**Key insight**: Model size scales linearly with n_mels due to Flatten+Dense bottleneck.

### 3. 13_tiny
| n_fft | n_mels | Accuracy | Size (KB) | Infer (ms) | AUC |
|-------|--------|----------|-----------|------------|------|
| 512 | 16 | 0.9160 | 5.91 | 0.04 | 0.9687 |
| 512 | 32 | 0.9366 | 8.66 | 0.07 | 0.9804 |
| 512 | 48 | 0.9332 | 11.41 | 0.11 | 0.9811 |
| 512 | 64 | 0.9354 | 14.16 | 0.14 | 0.9815 |
| 512 | 80 | 0.9382 | 16.91 | 0.17 | 0.9825 |
| 1024 | 16 | 0.9238 | 5.91 | 0.04 | 0.9742 |
| 1024 | 32 | 0.9364 | 8.66 | 0.07 | 0.9793 |
| 1024 | **48** | **0.9384** | 11.41 | 0.10 | 0.9803 |
| 1024 | 64 | 0.9366 | 14.16 | 0.14 | 0.9829 |
| 1024 | 80 | 0.9340 | 16.91 | 0.18 | 0.9798 |

### 4. 2_depthwise (Underperforms)
| n_fft | n_mels | Accuracy | Size (KB) | Infer (ms) | AUC |
|-------|--------|----------|-----------|------------|------|
| 512 | 80 | 0.9262 | 30.65 | 0.39 | 0.9733 |
| 1024 | 48 | 0.9254 | 19.65 | 0.24 | 0.9757 |

**Key insight**: SeparableConv2D lacks representational power for this task.

---

## Architecture Comparison

```
Architecture       Best Acc   Size (KB)   Key Feature              Params
─────────────────────────────────────────────────────────────────────────────
7_low_power        94.72%     5.12        GlobalAveragePooling     1,282
1_baseline         93.88%     29.28       Flatten+Dense            25,558
13_tiny            93.84%     11.41       Smaller input (96x48)    7,254
2_depthwise        92.62%     30.65       SeparableConv2D          25,443
```

---

## Why 7_low_power Wins

### Model Architecture Comparison

**7_low_power (1,282 params, 5.12 KB)**:
```
Input (184, 80, 1)
    ↓
Conv2D(8, 3x3, same, relu)  → 80 params
MaxPooling2D(2,2)           → (92, 40, 8)
    ↓
Conv2D(16, 3x3, same, relu) → 1,168 params
GlobalAveragePooling2D      → (16,)
    ↓
Dense(2, softmax)           → 34 params
```

**1_baseline (25,558 params, 29.28 KB)**:
```
Input (184, 80, 1)
    ↓
Conv2D(4, 3x3, valid, relu) → 40 params
MaxPooling2D(2,2)           → (91, 39, 4)
    ↓
Conv2D(4, 3x3, valid, relu) → 148 params
MaxPooling2D(2,2)           → (44, 18, 4)
    ↓
Flatten                     → (3168,)
Dense(8, relu)              → 25,352 params  ← BOTTLENECK!
Dense(2, softmax)           → 18 params
```

### Key Differences:
1. **GlobalAveragePooling vs Flatten**: 7_low_power eliminates the huge Dense layer
2. **More filters**: 8→16 vs 4→4 provides more representational capacity
3. **Same padding**: Preserves spatial information better than valid padding
4. **Translation invariance**: GlobalAveragePooling provides natural shift invariance

---

## Plan to Increase Accuracy

### Tier 1: High-Impact Changes (Expected +1-3% accuracy)

#### 1. Deeper GlobalAvgPool Architecture
```python
def build_deeper_gap(input_shape=(184, 80, 1), num_classes=2):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Block 1
    x = Conv2D(16, (3,3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 2
    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)

    # Block 3
    x = Conv2D(64, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
```
**Expected params**: ~20K, **Expected size**: ~20 KB

#### 2. Residual GlobalAvgPool Architecture
```python
def build_residual_gap(input_shape=(184, 80, 1), num_classes=2):
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Initial conv
    x = Conv2D(16, (3,3), padding='same', activation='relu')(inputs)
    x = MaxPooling2D((2,2))(x)

    # Residual block 1
    shortcut = x
    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(16, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)
    x = MaxPooling2D((2,2))(x)

    # Residual block 2
    shortcut = Conv2D(32, (1,1))(x)  # Project shortcut
    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32, (3,3), padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([shortcut, x])
    x = Activation('relu')(x)

    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs, outputs)
```
**Expected params**: ~15K, **Expected size**: ~15 KB

#### 3. Squeeze-and-Excitation Block
```python
def se_block(x, ratio=4):
    filters = x.shape[-1]
    se = GlobalAveragePooling2D()(x)
    se = Dense(filters // ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])
```

### Tier 2: Medium-Impact Changes (Expected +0.5-1% accuracy)

#### 4. Data Augmentation Enhancement
```python
def enhanced_augment_mel(mel, label):
    # Time masking
    time_mask_width = tf.random.uniform([], 0, 20, dtype=tf.int32)
    time_start = tf.random.uniform([], 0, 184 - 20, dtype=tf.int32)
    mel = tf.concat([
        mel[:time_start, :, :],
        tf.zeros([time_mask_width, mel.shape[1], 1]),
        mel[time_start + time_mask_width:, :, :]
    ], axis=0)

    # Frequency masking
    freq_mask_width = tf.random.uniform([], 0, 10, dtype=tf.int32)
    freq_start = tf.random.uniform([], 0, n_mels - 10, dtype=tf.int32)
    mel = tf.concat([
        mel[:, :freq_start, :],
        tf.zeros([mel.shape[0], freq_mask_width, 1]),
        mel[:, freq_start + freq_mask_width:, :]
    ], axis=1)

    # Gaussian noise
    noise = tf.random.normal(tf.shape(mel), mean=0.0, stddev=0.02)
    mel = mel + noise

    return tf.clip_by_value(mel, 0.0, 1.0), label
```

#### 5. Label Smoothing
```python
model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=[...]
)
```

#### 6. Mixup Augmentation
```python
def mixup(x1, y1, x2, y2, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    x = lam * x1 + (1 - lam) * x2
    y = lam * y1 + (1 - lam) * y2
    return x, y
```

### Tier 3: Experimental Changes (Variable impact)

#### 7. Attention Mechanism
```python
def channel_attention(x):
    # Simple channel attention
    gap = GlobalAveragePooling2D()(x)
    gmp = GlobalMaxPooling2D()(x)
    attention = Dense(x.shape[-1] // 4, activation='relu')(gap + gmp)
    attention = Dense(x.shape[-1], activation='sigmoid')(attention)
    return x * attention[:, tf.newaxis, tf.newaxis, :]
```

#### 8. Multi-Scale Features
```python
def multi_scale_block(x):
    branch1 = Conv2D(8, (1,1), padding='same', activation='relu')(x)
    branch3 = Conv2D(8, (3,3), padding='same', activation='relu')(x)
    branch5 = Conv2D(8, (5,5), padding='same', activation='relu')(x)
    return Concatenate()([branch1, branch3, branch5])
```

---

## Recommended Experimental Plan

### Phase 1: Quick Wins
1. **14_deeper_gap.py**: 3-block GlobalAvgPool architecture with BatchNorm
2. **15_gap_bn.py**: 7_low_power + BatchNormalization
3. **16_gap_dropout.py**: 7_low_power + Dropout(0.2) before final Dense

### Phase 2: Architecture Exploration
4. **17_residual_gap.py**: Residual connections + GlobalAvgPool
5. **18_se_gap.py**: Squeeze-and-Excitation + GlobalAvgPool
6. **19_attention_gap.py**: Channel attention + GlobalAvgPool

### Phase 3: Training Improvements
7. Test with enhanced data augmentation (SpecAugment)
8. Test with label smoothing (0.1)
9. Test with mixup augmentation

### Phase 4: Ensemble
10. Ensemble of best 3-5 models for maximum accuracy

---

## Expected Outcomes

| Architecture | Expected Acc | Expected Size | Risk |
|--------------|-------------|---------------|------|
| 14_deeper_gap | 95-96% | ~20 KB | Low |
| 15_gap_bn | 95-96% | ~6 KB | Low |
| 17_residual_gap | 95-97% | ~15 KB | Medium |
| 18_se_gap | 95-96% | ~8 KB | Medium |
| Ensemble (top 3) | 96-97% | N/A | Low |

---

## Conclusions

1. **GlobalAveragePooling is superior** to Flatten+Dense for this task
2. **Higher n_mels (80) helps** when model can handle the capacity
3. **SeparableConv2D underperforms** - standard Conv2D is better
4. **Model size and accuracy can be decoupled** with GlobalAvgPool

The path to higher accuracy lies in:
- Deeper architectures with GlobalAveragePooling
- Adding BatchNormalization for training stability
- Residual connections for gradient flow
- Attention mechanisms for feature refinement
- Enhanced data augmentation

Current best: **94.72% @ 5.12 KB** (7_low_power, n_fft=1024, n_mels=80)
Target: **96-97% @ <25 KB**
