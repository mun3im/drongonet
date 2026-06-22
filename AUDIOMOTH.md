---
name: audiomoth-hardware-specs
description: AudioMoth ARM Cortex-M4F hardware constraints and energy analysis
metadata:
  type: reference
---

# AudioMoth Hardware Specifications

## Overview

AudioMoth is deployed on the **Silicon Labs EFM32 Wonder Gecko (ARM Cortex-M4F)**, the most widely deployed Autonomous Recording Unit (ARU) platform. These specs determine the size and latency budgets for SEABADNet deployments.

**Official Manufacturer Sources:**
- [Open Acoustic Devices Datasheets](https://www.openacousticdevices.info/datasheets)
- [AudioMoth: A Low-Cost Acoustic Device (HardwareX)](https://www.hardware-x.com/article/S2468-0672(19)30030-6/fulltext)
- Benhammadi et al. (2026) — measured energy on AudioMoth hardware

## MCU & Memory Constraints

| Parameter | Value | Notes |
|-----------|-------|-------|
| **MCU** | Silicon Labs EFM32 Wonder Gecko (ARM Cortex-M4F) | The most widely deployed ARU platform |
| **Clock speed** | 48 MHz | Much lower than generic Cortex-M4 (which often runs 80–240 MHz) |
| **Flash** | 256 KB | Limited for large models |
| **Internal SRAM** | 32 KB | **Critical constraint for tensor arena** |
| **External SRAM** | 256 KB (dev variant only) | Standard AudioMoth has internal 32 KB only |

## Power Consumption

| Metric | Range | Notes |
|--------|-------|-------|
| **Recording current (no SD)** | 19–23 mA | Baseline continuous recording |
| **Recording current (with SD writes)** | 10–40 mA | Varies by SD card model/speed (Benhammadi: 11.87 mA typical with writes) |
| **Sleep/Idle current** | 30–65 µA | Firmware-dependent |
| **Operating voltage** | 3.1–20 V (typical 3.6–6 V) | AA batteries → 3.6–4.8 V typical |

## Audio Interface

| Parameter | Value |
|-----------|-------|
| **Audio sampling rate** | 20,480 Hz or 16 kHz (SEABAD configuration) |
| **Standard ARU configuration** | 16 kHz, 3-second clips |

## Detection Algorithm Energy Comparison (Benhammadi et al. 2026, Table 4)

Comparison of detection algorithms on AudioMoth hardware. **Key insight:** Goertzel (industry baseline) has similar average current to ML-based Mel detector (11.87 vs 12.04 mA), but produces many more false positives (p=0.63 vs 0.09). These false positives trigger SD card writes, which consume significant energy, reducing effective battery life from 221h to 216h.

| Detector | Always-On Current | Detection Current | Hit Rate | Avg Current | Battery Life | Notes |
|----------|-------------------|------------------|----------|------------|--------------|-------|
| **Goertzel** (industry baseline) | 9.27 mA @ 810 ms | 64 mA @ 60 ms | 0.63 | **11.87 mA** | **221 h ± 2 min** | Many false positives → more SD writes |
| **Mel** (CNN-based) | 11.45 mA @ 1500 ms | 18.20 mA @ 1650 ms | 0.09 | **12.04 mA** | **216 h ± 3 min** | Fewer false positives, slightly lower battery |
| **Gabor** (Learnable filters) | 9.65 mA @ 1500 ms | 18.4 mA @ 1670 ms | 0.14 | **10.97 mA** | **241 h ± 5 min** | **Best battery life** despite higher per-detection current |
| **Band** (Amplitude envelope) | 9.70 mA @ 1500 ms | 18.4 mA @ 1620 ms | 0.23 | **11.43 mA** | **228 h ± 5 min** | — |

### Key Finding

Goertzel's apparent efficiency is misleading. While its average current (11.87 mA) is competitive with ML methods, it produces 7× more false positives (0.63 hit rate), which dramatically increases SD card write energy. This demonstrates that ML-based detection achieves higher autonomy despite marginally higher per-detection overhead by drastically reducing false positives.

## Implications for SEABADNet

- **Internal SRAM (32 KB) is the binding constraint** for AudioMoth deployment
  - SEABADNet-Micro's tensor arena = ~23 KB measured (safe margin)
  - SEABADNet-Edge requires SBC deployment (Raspberry Pi, Portenta X8)
  
- **Clock speed (48 MHz) is 2.5–5× slower than desktop CPUs**
  - Latency targets: ≤1 ms for Micro, ≤2 ms for Edge
  - Depthwise separable convolutions preferred to reduce MAC count
  
- **False positive rate directly impacts battery life**
  - Reducing false positives from p=0.63 to p=0.09 (7× improvement) yields 5-hour battery extension
  - Recall optimization is as important as inference latency
