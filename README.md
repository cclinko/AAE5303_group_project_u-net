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

## 📑 Table of Contents
1. [Project Summary](#project-summary)
2. [Introduction](#introduction)
3. [Dataset Description](#dataset-description)
4. [Methodology](#methodology)
5. [Implementation Details](#implementation-details)
6. [Results and Analysis](#results-and-analysis)
7. [Visualization](#visualization)
8. [Discussion](#discussion)
9. [Future Outlook](#future-outlook)

---

<span id="project-summary"></span>
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

<span id="introduction"></span>
## 💡 Introduction
With the explosive growth of the low-altitude economy, autonomous navigation and safe obstacle avoidance for UAVs in complex urban environments have become core challenges for UTM (UAS Traffic Management) systems. This project utilizes the **U-Net** architecture to provide "pixel-level" environmental perception.

The system supports:
* **Route Planning**: Precise extraction of building boundaries to construct safe 3D flight corridors.
* **Emergency Response**: Real-time identification of paved roads or flat green fields for emergency landings.
* **Micro-obstacle Avoidance**: Accurate detection of solar boards, vehicles, and other fatal micro-hazards.

---

<span id="dataset-description"></span>
## 🗺️ Dataset Description
We used **AMtown Dataset**, which features complex urban topologies and highly imbalanced class distributions.
* **Sources**: AMtown01, AMtown02, AMtown03.
* **Split Strategy**:
  * **Training Set**: AMtown01 + AMtown02.
  * **Test Set**: AMtown03 (447 physical image frames).
* **Task Definition**: Pixel-level semantic segmentation of 26 distinct categories.
* **Core Challenge**: Extreme class imbalance where background and green fields overwhelm classes like vehicles and solar boards.

---

<span id="methodology"></span>
## 🧠 Methodology

To tackle the complexities of urban terrain and the severe class imbalance in the AMtown dataset, we designed a pipeline that prioritizes geometric recovery and mathematically penalizes long-tail dominance.

### 1. Base Architecture: U-Net Trained from Scratch
We adopted the classic **U-Net** architecture, leveraging its symmetric encoder-decoder structure and skip connections to recover fine-grained spatial geometries and sharp building edges. 
* **Zero Pre-training:** Instead of using pre-trained backbones (e.g., ResNet on ImageNet), we trained the model entirely from scratch. This ensures the convolutional filters are strictly optimized for the unique top-down perspective and specific structural textures of the UAV aerial dataset, completely avoiding domain-shift biases.

### 2. Strategy I: Dynamic Class Weights (Long-tail Optimization)
The AMtown dataset exhibits an extreme long-tail distribution. To prevent the model from collapsing into a "greedy prediction" state, we implemented a custom-weighted `CrossEntropyLoss`. By hardcoding physical priorities into the loss function, we reshaped the model's optimization trajectory:
* **Suppressing Dominant Classes:** Scaled down weights for massive regions like `background` (0.1) and `green_field` (0.5) to strictly penalize lazy, high-frequency predictions.
* **Boosting Structural Boundaries:** Applied a **5x multiplier** to linear infrastructure like `paved_motor_road` and `dirt_motor_road` to ensure continuous and unbroken route extraction.
* **Protecting Micro-objects:** Assigned a massive **10x multiplier** to highly vulnerable and sparse categories (`solar_board`, `vehicles`). This forces the gradients to update aggressively when these critical micro-obstacles are misclassified, guaranteeing their visibility in the final cognitive map.

### 3. Strategy II: Online Data Augmentation (Spatial Memory Disruption)
Deep neural networks can easily overfit to the absolute spatial coordinates of a static dataset (e.g., memorizing specific map layouts). To disrupt this "spatial memory," we integrated dynamic geometric perturbations within the PyTorch DataLoader:
* **Synchronous Transformations:** We applied random horizontal and vertical flips with a 50% probability to both the input images and their corresponding ground-truth masks simultaneously on the fly. 
* **Topological Learning:** This intervention effectively strips the model of its spatial coordinate memory, forcing the convolutional kernels to learn robust, orientation-invariant topological features of the urban structures.

---

<span id="implementation-details"></span>
## 💻 Implementation Details
* **Framework**: Google Colab / PyTorch.
* **Hyperparameters**:
  * **Learning Rate**: Recalibrated to **1e-6** to prevent gradient explosion (NaN) observed at 1e-4.
  * **Scale**: Global downscaling factor of 0.25.
* **Mixed Precision Training (AMP)**: Implemented using `torch.amp.autocast('cuda')` for faster training and reduced VRAM usage.

---

<span id="results-and-analysis"></span>
## 📊 Results and Analysis

### Performance Evolution
* **Baseline**: Suffered from a greedy strategy, predicting almost everything as background; micro-object scores were zero.
* **Optimized**: Achieved valid scores across all critical categories including roads, vehicles, and bridges.

### Training Dynamics
Over 30 epochs, training loss converged from 1.95 to 0.136. Validation metrics showed stable upward trends without overfitting.

---

<span id="visualization"></span>
## 🎨 Visualization
* **Metrics Visualization**: [Insert Bar Chart of Per-Class IoU]
* **Prediction Comparison**: [Insert Ground Truth vs. Prediction Comparison]
> **Observation**: The optimized model successfully restored building edges and recovered "invisible" micro-objects that were completely missed in the baseline.

---

<span id="discussion"></span>
## 🤔 Discussion

This project highlights a critical reality in deploying deep learning models for real-world robotics: **algorithmic architecture alone cannot overcome fundamental physical and computational constraints.**

Due to severe hardware limitations, the required **0.25** global downscaling drastically compressed the already scarce pixel footprint of critical micro-objects. Consequently, the baseline U-Net fell into a classic **"greedy prediction trap"**—achieving superficially high pixel accuracy by aggressively predicting dominant background classes while completely blinding itself to long-tail categories.

To break this bottleneck, we shifted our focus from architectural changes to **Engineering Intervention**, fundamentally altering the model's learning behavior:

* **Redefining Value (Class Weights):** We explicitly reweighted the loss function based on physical safety importance rather than pixel frequency, forcing the model to prioritize critical micro-obstacles.
* **Disrupting Spatial Memory (Data Augmentation):** We introduced dynamic geometric flips to strip the model of its spatial coordinate memory, compelling it to learn true urban topological features.

Ultimately, this project taught us that throwing a better network architecture at a problem isn't always the answer, especially under strict hardware constraints. The real breakthrough came from looking at the data through an engineering lens—realizing that a few pixels of a solar panel matter far more than thousands of pixels of grass. It was this human intuition, explicitly hardcoded into the training pipeline, that made the system actually work in a practical scenario.

---

<span id="future-outlook"></span>
## 🚀 Future Outlook

To support advanced Vision-and-Language Navigation (VLN) and robust UAV obstacle avoidance, the system must evolve in terms of resolution, boundary perception, and 3D integration. Below is our engineering evaluation matrix for the next technical iterations:

| Optimization Strategy | Engineering Trade-offs (Pros & Cons) | Implementation Cost | ROI / Priority |
| :--- | :--- | :---: | :---: |
| **1. HD Patch-based Training**<br>*(Random 512x512 crops replacing global downscaling)* | **Pros:** Fundamentally resolves the loss of micro-features caused by image compression.<br>**Cons:** Drastically increases training epochs and requires complex sliding-window logic for inference. | ⏱️ Medium | ⭐⭐⭐⭐ |
| **2. Focal Loss Integration**<br>*(Hard sample mining)* | **Pros:** Forces the model to focus on ambiguous boundaries with **zero** added inference latency.<br>**Cons:** Highly sensitive to hyperparameter tuning (Gamma) and noisy dataset labels. | ⏱️ Low | ⭐⭐⭐⭐⭐ |
| **3. True 3D Semantic Fusion**<br>*(Back-projecting 2D masks onto 3D Gaussian Splats)* | **Pros:** Builds the ultimate Cognitive Map for UAVs, perfectly bridging 2D semantics with 3D geometry.<br>**Cons:** Extreme mathematical complexity; requires rigorous pixel-level alignment between 2D inference and 3D camera poses. | ⏱️ Very High | ⭐⭐⭐ |
