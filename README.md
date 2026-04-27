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

### 🏆 Key Results
| Metrics | Optimized Model |
| :--- | :---: |
| **Pixel Accuracy** | **92.20%** |
| **Mean IoU** | **60.59%** |
| **Mean Dice** | **69.95%** |

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

To tackle the severe class imbalance and geometric complexities of the AMtown dataset, we designed an optimization pipeline that mathematically prioritizes micro-features over dominant backgrounds.

### 1. Base Architecture: U-Net from Scratch
We adopted a classic **U-Net** trained entirely from scratch, leveraging its symmetric encoder-decoder structure to recover fine-grained spatial geometries.

> **Architectural Philosophy:** By strictly avoiding ImageNet pre-trained backbones, we ensured the convolutional filters were optimized *only* for the unique top-down perspective and structural textures of UAV aerial imagery, thereby eliminating domain-shift biases.

### 2. Strategy I: Dynamic Class Weights
The dataset exhibits an extreme long-tail distribution. To prevent the model from collapsing into a "greedy prediction" state (predicting everything as background), we hardcoded physical priorities into our `CrossEntropyLoss`. 

Here is our explicitly defined value system:

| Target Class Category | Weight Multiplier | Engineering Purpose |
| :--- | :---: | :--- |
| `background`, `green_field` | **⬇️ 0.1x / 0.5x** | **Suppress Dominant Classes**: Strictly penalize lazy, high-frequency background predictions. |
| `paved_motor_road`, `dirt_road` | **⬆️ 5.0x** | **Boost Structural Boundaries**: Ensure continuous and unbroken route extraction. |
| `solar_board`, `vehicles` | **🚀 10.0x** | **Protect Micro-objects**: Force aggressive gradient updates for critical UAV obstacles. |

> **The Logic:** We suppressed massive regions (background, green fields) while injecting weight multipliers (up to 10x) for vulnerable micro-objects (solar boards, vehicles).
> 
> <img src="image/图片0.png" width="600" alt="CrossEntropyLoss with Class Weights">

### 3. Strategy II: Online Data Augmentation
Deep neural networks can easily overfit to the absolute spatial coordinates of a static dataset. To disrupt this "spatial memory," we integrated dynamic geometric perturbations.

* 🔄 **The Action (Synchronous Flips):** Random horizontal and vertical flips (50% probability) applied synchronously to both the input images and their ground-truth masks on the fly during data loading.
* 🎯 **The Purpose (Topological Learning):** This intervention effectively strips the model of its spatial coordinate memory. Instead of memorizing specific map layouts, the convolutional kernels are forced to learn robust, orientation-invariant topological features of the urban structures.

<img src="image/图片1.png" width="400" alt="Online Data Augmentation Code">

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

Our analysis evaluates the model's ability to overcome the "greedy prediction" trap and successfully recover micro-objects under computational constraints. We present both quantitative metric improvements and the actual semantic outputs generated by the models.

### 1. Quantitative Performance Evolution
Following our dynamic weight interventions, the model successfully activated on minority classes and achieved a massive leap in core overall metrics.

| Evaluation Dimension | Baseline (Greedy) | Optimized (Weighted) | Improvement (Gain) |
| :--- | :---: | :---: | :--- |
| **Pixel Accuracy** | 85.53% | **92.20%** | 🚀 + 6.67% |
| **Mean IoU** | 26.54% | **60.59%** | 🚀 **+ 34.05%** |
| **Mean Dice** | 32.75% | **69.95%** | 🚀 **+ 37.20%** |

**The Micro-Object Breakthrough:** The most significant achievement was resurrecting the detection of fatal UAV obstacles that previously scored zero:
*  **Solar Board:** IoU jumped from `0.0000` ➔ **`0.6668`**
*  **Bridge:** IoU jumped from `0.0000` ➔ **`0.6669`**
*  **Vehicles (Sedan):** IoU jumped from `0.0000` ➔ **`0.4341`**

### 2. Output Analysis: Overcoming the Baseline Failure
To understand the metric improvements, we must analyze the physical semantic outputs. 

**The Baseline Failure:** Before optimization, the naive network aggressively predicted almost the entire map as a single class (green field), completely ignoring the underlying urban topology.
> ![Baseline Prediction Failure](image/a.png)
> *(Observation: Critical architectural structures and roads are entirely swallowed by false-positive field predictions.)*

**The Optimized Output:** After injecting physical priorities, the network was forced to respect the true geometric boundaries of the environment.
> ![Optimized Output Success](image/b.png)
> *(Observation: Building footprints are now sharply delineated, road networks are continuous, and micro-obstacles are successfully detected.)*

### 📦 Pre-trained Models
The trained model weights (`.pth`) are hosted on Google Drive due to file size limits:
* [Download checkpoint_epoch30.pth (Google Drive)]([https://drive.google.com/file/d/16YivEBp0F93EYIbe_kAPZYqTsg2YxQsP/view?usp=sharing])

---

<span id="visualization"></span>
## 📈 Data Visualization

This section provides the data visualization of our training dynamics and the per-class metric distribution, contrasting the baseline with the optimized pipeline.

### 1. Per-Class Metric Distribution (IoU & Dice)
The bar charts below illustrate the dramatic shift from a highly skewed, long-tail failure to a balanced, comprehensive cognitive map.

**Baseline Distribution:**
> ![Baseline Bar Chart](image/a2.png)
> *(Note the complete failure—scores of 0.0000—across more than half of the categories, including all micro-objects.)*

**Optimized Distribution:**
> ![Optimized Bar Chart](image/b2.png)
> *(The optimized model successfully establishes valid, high-scoring statistical boundaries across all 26 complex urban categories.)*

### 2. Training Dynamics & Convergence
We visualized the Loss trajectory and Validation metrics to confirm model stability and the absence of overfitting.

**Baseline Convergence (5 Epochs):**
> ![Baseline Training Curve](image/a3.png)

**Optimized Convergence (30 Epochs):**
> ![Optimized Training Curve](image/b3.png)
> *(The optimized training loss smoothly converged to 0.1363, while the Validation Dice score maintained a consistent upward trend, peaking at 0.9597.)*

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
