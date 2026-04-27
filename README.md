# Urban Terrain Semantic Segmentation and Optimization based on U-Net

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
[cite_start]This project addresses the environmental perception needs of low-altitude Unmanned Aerial Vehicles (UAVs) in complex urban terrains[cite: 20, 21]. [cite_start]Under constraints of limited computational power (Global Scaling = 0.25) and extreme class imbalance (Long-tail distribution), we trained and optimized a U-Net semantic segmentation model[cite: 330].

[cite_start]By introducing **Customized Dynamic Class Weight Penalties** and **Online Data Augmentation**, we successfully resolved the local optimum trap where the model failed to predict minority classes[cite: 42, 161]. [cite_start]The final model achieved a significant leap in core evaluation metrics, particularly for micro-objects[cite: 229].

### 🏆 Key Results Comparison
| Metrics | Baseline Model | Optimized Model | Improvement (Gain) |
| :--- | :---: | :---: | :---: |
| **Pixel Accuracy** | [cite_start]85.50% [cite: 57] | [cite_start]**92.20%** [cite: 228] | 🚀 + 6.70% |
| **Mean IoU** | [cite_start]24.80% [cite: 61] | [cite_start]**60.59%** [cite: 229] | 🚀 **+ 35.79%** |
| **Mean Dice** | [cite_start]32.75% [cite: 100] | [cite_start]**69.95%** [cite: 229] | 🚀 **+ 37.20%** |

[cite_start]*Note: The peak Validation Dice (excluding background) stabilized at **0.9597**[cite: 230].*

---

## 🌟 Introduction
[cite_start]With the explosive growth of the low-altitude economy, autonomous navigation and safe obstacle avoidance for UAVs in complex urban environments have become core challenges for UTM (UAS Traffic Management) systems[cite: 20]. [cite_start]This project utilizes the **U-Net** architecture to provide "pixel-level" environmental perception[cite: 21, 45].

The system supports:
* [cite_start]**Route Planning**: Precise extraction of building boundaries to construct safe 3D flight corridors[cite: 25].
* [cite_start]**Emergency Response**: Real-time identification of paved roads or flat green fields for emergency landings[cite: 26].
* [cite_start]**Micro-obstacle Avoidance**: Accurate detection of solar boards, vehicles, and other fatal micro-hazards[cite: 27, 28].

---

## 📊 Dataset Description
[cite_start]We used the customized **AMtown Dataset**, which features complex urban topologies and highly imbalanced class distributions[cite: 31].
* [cite_start]**Sources**: AMtown01, AMtown02, AMtown03[cite: 31].
* **Split Strategy**:
  * [cite_start]**Training Set**: AMtown01 + AMtown02[cite: 31].
  * **Test Set**: AMtown03 (447 physical image frames).
* [cite_start]**Task Definition**: Pixel-level semantic segmentation of 26 distinct categories[cite: 30].
* [cite_start]**Core Challenge**: Extreme class imbalance where background and green fields overwhelm classes like vehicles and solar boards[cite: 32].

---

## 🧠 Methodology

### 1. Base Architecture
[cite_start]A classic **U-Net** architecture trained from scratch (without pre-trained weights)[cite: 45].

### 2. Strategy I: Dynamic Class Weights
[cite_start]To mathematically penalize the long-tail effect, we introduced a weighted `CrossEntropyLoss`[cite: 161]:
* [cite_start]**Suppress Dominant Classes**: Reduced weights for `background` (0.1) and `green_field` (0.5) to prevent model "laziness"[cite: 171].
* [cite_start]**Boost Minority Classes**: 5x weight multiplier for roads[cite: 173].
* [cite_start]**Protect Micro-objects**: 10x weight multiplier for highly vulnerable categories like `solar_board` and `vehicles`[cite: 175, 176].

### 3. Strategy II: Online Data Augmentation
[cite_start]Forces the model to learn true physical topology rather than memorizing spatial coordinates[cite: 201]:
* [cite_start]Synchronous random horizontal and vertical flips applied with a 50% probability during data loading[cite: 193].

---

## ⚙️ Implementation Details
* **Framework**: Google Colab / PyTorch.
* **Hyperparameters**:
  * [cite_start]**Learning Rate**: Recalibrated to **1e-6** to prevent gradient explosion (NaN) observed at 1e-4[cite: 46, 47].
  * [cite_start]**Scale**: Global downscaling factor of 0.25[cite: 330].
* [cite_start]**Mixed Precision Training (AMP)**: Implemented using `torch.amp.autocast('cuda')` for faster training and reduced VRAM usage[cite: 92, 95].

---

## 📈 Results and Analysis

### Performance Evolution
* [cite_start]**Baseline**: Suffered from a greedy strategy, predicting almost everything as background; micro-object scores were zero[cite: 42].
* [cite_start]**Optimized**: Achieved valid scores across all critical categories including roads, vehicles, and bridges[cite: 229].

### Training Dynamics
[cite_start]Over 30 epochs, training loss converged from 1.95 to 0.136[cite: 219, 225]. [cite_start]Validation metrics showed stable upward trends without overfitting[cite: 230].

---

## 🖼️ Visualization
* **Metrics Visualization**: [Insert Bar Chart of Per-Class IoU]
* **Prediction Comparison**: [Insert Ground Truth vs. Prediction Comparison]
> **Observation**: The optimized model successfully restored building edges and recovered "invisible" micro-objects that were completely missed in the baseline.

---

## 💬 Discussion
[cite_start]This project demonstrates that under computational constraints (0.25 scaling), architectural selection alone is insufficient[cite: 330]. **Engineering intervention**—specifically human-defined physical importance via class weight penalties—is the key to breaking performance bottlenecks in imbalanced datasets.

---

## 🚀 Future Outlook
[cite_start]To support advanced Vision-and-Language Navigation (VLN), the system will evolve in three directions[cite: 331]:

1. [cite_start]**Patch-based Training**: Discard global downscaling in favor of random 512x512 HD patches to improve resolution of micro-features[cite: 332, 333].
2. [cite_start]**Advanced Loss Functions**: Introduce **Focal Loss** for "hard sample mining" to focus on ambiguous boundaries[cite: 335, 336].
3. [cite_start]**True 3D Semantic Fusion**: Back-project 2D masks onto 3D Gaussian splats via camera intrinsics to build seamless 3D Semantic Cognitive Maps[cite: 337, 362].
