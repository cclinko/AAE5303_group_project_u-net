<div align="center">

# AAE5303 Assignment: Semantic Segmentation with U-Net

<img src="https://img.shields.io/badge/NETWORK-U--NET-1f77b4?style=for-the-badge" alt="Network">
<img src="https://img.shields.io/badge/MODE-SEMANTIC_SEGMENTATION-9bc53d?style=for-the-badge" alt="Mode">
<img src="https://img.shields.io/badge/DATASET-AMtown-f26419?style=for-the-badge" alt="Dataset">
<img src="https://img.shields.io/badge/STATUS-OPTIMIZED-5cb85c?style=for-the-badge" alt="Status">

<h3>Pixel-Level Environmental Perception on UAV Aerial Imagery</h3>

<i>AMtown Dataset - 26 Categories Classification</i>

<br>
</div>

## 📚 Table of Contents
1. [Project Summary](#-project-summary)
2. [Introduction](#-introduction)
3. [Dataset Description](#-dataset-description)
4. [Methodology](#-methodology)
5. [Implementation Details](#-implementation-details)
6. [Results and Analysis](#-results-and-analysis)
7. [Visualization](#-visualization)
8. [Discussion](#-discussion)
9. [Future Outlook](#-future-outlook)

---

## 📝 Project Summary
This project addresses the environmental perception needs of low-altitude Unmanned Aerial Vehicles (UAVs) in complex urban terrains. Under constraints of limited computational power (Global Scaling = 0.25) and extreme class imbalance (Long-tail distribution), we trained and optimized a U-Net semantic segmentation model.

By introducing **Customized Dynamic Class Weight Penalties** and **Online Data Augmentation**, we successfully resolved the local optimum trap where the model failed to predict minority classes. The final model achieved a significant leap in core evaluation metrics, particularly for micro-objects.

### 🏆 Key Results Comparison
| Metrics | Baseline Model | Optimized Model | Improvement (Gain) |
| :--- | :---: | :---: | :---: |
| **Pixel Accuracy** | 85.50% | **92.20%** | 🚀 + 6.70% |
| **Mean IoU** | 24.80% | **60.59%** | 🚀 **+ 35.79%** |
| **Mean Dice** | 32.75% | **69.95%** | 🚀 **+ 37.20%** |

*Note: The peak Validation Dice (excluding background) stabilized at **0.9597**.*

---

## 🌟 Introduction
With the explosive growth of the low-altitude economy, autonomous navigation and safe obstacle avoidance for UAVs in complex urban environments have become core challenges for UTM (UAS Traffic Management) systems. This project utilizes the **U-Net** architecture to provide "pixel-level" environmental perception.

The system supports:
* **Route Planning**: Precise extraction of building boundaries to construct safe 3D flight corridors.
* **Emergency Response**: Real-time identification of paved roads or flat green fields for emergency landings.
* **Micro-obstacle Avoidance**: Accurate detection of solar boards, vehicles, and other fatal micro-hazards.

---

## 📊 Dataset Description
We used the customized **AMtown Dataset**, which features complex urban topologies and highly imbalanced class distributions.
* **Sources**: AMtown01, AMtown02, AMtown03.
* **Split Strategy**:
  * **Training Set**: AMtown01 + AMtown02.
  * **Test Set**: AMtown03 (447 physical image frames).
* **Task Definition**: Pixel-level semantic segmentation of 26 distinct categories.
* **Core Challenge**: Extreme class imbalance where background and green fields overwhelm classes like vehicles and solar boards.

---

## 🧠 Methodology

### 1. Base Architecture
A classic **U-Net** architecture trained from scratch (without pre-trained weights).

### 2. Strategy I: Dynamic Class Weights
To mathematically penalize the long-tail effect, we introduced a weighted `CrossEntropyLoss`:
* **Suppress Dominant Classes**: Reduced weights for `background` (0.1) and `green_field` (0.5) to prevent model "laziness".
* **Boost Minority Classes**: 5x weight multiplier for roads.
* **Protect Micro-objects**: 10x weight multiplier for highly vulnerable categories like `solar_board` and `vehicles`.

### 3. Strategy II: Online Data Augmentation
Forces the model to learn true physical topology rather than memorizing spatial coordinates:
* Synchronous random horizontal and vertical flips applied with a 50% probability during data loading.

---

## ⚙️ Implementation Details
* **Framework**: Google Colab / PyTorch.
* **Hyperparameters**:
  * **Learning Rate**: Recalibrated to **1e-6** to prevent gradient explosion (NaN) observed at 1e-4.
  * **Scale**: Global downscaling factor of 0.25.
* **Mixed Precision Training (AMP)**: Implemented using `torch.amp.autocast('cuda')` for faster training and reduced VRAM usage.

---

## 📈 Results and Analysis

### Performance Evolution
* **Baseline**: Suffered from a greedy strategy, predicting almost everything as background; micro-object scores were zero.
* **Optimized**: Achieved valid scores across all critical categories including roads, vehicles, and bridges.

### Training Dynamics
Over 30 epochs, training loss converged from 1.95 to 0.136. Validation metrics showed stable upward trends without overfitting.

---

## 🖼️ Visualization
* **Metrics Visualization**: [Insert Bar Chart of Per-Class IoU]
* **Prediction Comparison**: [Insert Ground Truth vs. Prediction Comparison]
> **Observation**: The optimized model successfully restored building edges and recovered "invisible" micro-objects that were completely missed in the baseline.

---

## 💬 Discussion
This project demonstrates that under computational constraints (0.25 scaling), architectural selection alone is insufficient. **Engineering intervention**—specifically human-defined physical importance via class weight penalties—is the key to breaking performance bottlenecks in imbalanced datasets.

---

## 🚀 Future Outlook
To support advanced Vision-and-Language Navigation (VLN), the system will evolve in three directions:

1. **Patch-based Training**: Discard global downscaling in favor of random 512x512 HD patches to improve resolution of micro-features.
2. **Advanced Loss Functions**: Introduce **Focal Loss** for "hard sample mining" to focus on ambiguous boundaries.
3. **True 3D Semantic Fusion**: Back-project 2D masks onto 3D Gaussian splats via camera intrinsics to build seamless 3D Semantic Cognitive Maps.
