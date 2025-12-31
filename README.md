# Everyday Objects Detection with YOLO

## 1. Introduction to Object Detection & YOLO

### What is Object Detection?
Object detection is a computer vision technology that allows machines to identify and locate objects within an image or video. Unlike simple image classification (which answers "what is in this image?"), object detection answers **"what is where?"** by drawing bounding boxes around detected objects and assigning them a class label.

### Traditional vs. Modern Approaches
*   **Old School (e.g., R-CNN, Fast R-CNN):** These were "two-stage" detectors.
    1.  **Stage 1:** Generate "region proposals" (potential areas where an object might be).
    2.  **Stage 2:** Classify the object in each proposed region.
    *   *Drawback:* This process is computationally expensive and slow, making real-time detection difficult.

*   **YOLO (You Only Look Once):** A "single-stage" detector.
    *   YOLO revolutionized the field by framing object detection as a **single regression problem**.
    *   It looks at the entire image once (hence the name) and predicts bounding boxes and class probabilities simultaneously.
    *   *Advantage:* Extremely fast and suitable for real-time applications like self-driving cars and video analytics.

### How YOLO Works
1.  **Grid System:** The input image is divided into an $S \times S$ grid.
2.  **Responsibility:** If the center of an object falls into a grid cell, that cell is responsible for detecting it.
3.  **Predictions:** Each grid cell predicts:
    *   **Bounding Boxes:** Coordinates $(x, y, w, h)$ for the object.
    *   **Confidence Score:** How likely it is that the box contains an object ($Pr(Object) \times IoU$).
    *   **Class Probabilities:** Which category the object belongs to (e.g., "Smart_phone").

### Base YOLO vs. Fast YOLO
In the original YOLO development, two main variations were introduced to balance speed and accuracy:
*   **Base YOLO:** The standard model designed for a balance of high accuracy and good speed. It uses a deeper Convolutional Neural Network (CNN) to extract rich features from images.
*   **Fast YOLO:** A lighter version designed for maximum speed. It uses a neural network with fewer convolutional layers and fewer filters. While slightly less accurate than the Base model, it is significantly faster, making it ideal for lower-power devices (like Raspberry Pi) or extremely high-frame-rate requirements.

---

## 2. Project Overview

**Goal:** The objective of this project is to build a custom object detection system capable of identifying five specific everyday objects in real-time.

### The Dataset
*   **Total Images:** 99 images collected under various lighting conditions.
*   **Classes (5):**
    1.  `Air_pod`
    2.  `Multi_adapter`
    3.  `Smart_phone`
    4.  `Water_bottle`
    5.  `world_globe`
*   **Labeling:** Images were manually annotated using **Label Studio**. Bounding boxes were drawn around each object, and annotations were exported in the **YOLO format** (text files containing normalized coordinates for each image).

---

## 3. Training Process

### Environment
The model was trained using **Google Colab**, leveraging a **Tesla T4 GPU** for accelerated processing.

### Model Architecture
We used **YOLO11s (Small)**. The "s" stands for small, which is a modern equivalent of the "Base" concept—powerful enough for high accuracy but light enough for efficient training and inference.

### Configuration (`data.yaml`)
The training is guided by a `.yaml` file, which acts as a map for the model. It defines:
*   **Path:** Where the dataset is located.
*   **Train/Val:** Which folders contain training vs. validation images.
*   **nc (Number of Classes):** 5.
*   **names:** The list of class names (Air_pod, etc.).

### Hyperparameters & Key Terms
We configured the training with the following parameters:

*   **Epochs (60):** An epoch is one complete pass of the entire training dataset through the neural network. We trained for 60 epochs to allow the model sufficient time to learn features without "overfitting" (memorizing) the data.
*   **Batch Size (16):** Instead of feeding all images at once, the model sees a "batch" of 16 images at a time. This updates the model weights more frequently and fits within GPU memory.
*   **Image Size (640):** All input images are resized to $640 \times 640$ pixels before processing to ensure consistency.

### Training Results
*   **mAP (Mean Average Precision):** This is the gold standard metric for object detection.
    *   **mAP@50:** We achieved a score of **~0.97 (97%)**. This means that when we require the predicted box to overlap at least 50% with the true box, the model is correct 97% of the time.
*   **Loss:** During training, both "Box Loss" (error in position) and "Class Loss" (error in identifying the object type) steadily decreased, indicating successful learning.

---

## 4. Local Deployment & Testing

After training, the best-performing model weights (`best.pt`) were downloaded to a local PC for testing.

### Requirements
To run the model locally, you need Python and the following libraries:
```bash
pip install ultralytics opencv-python numpy
```

### Running the Detector
We use the `yolo_detect.py` script to run inference. This script can handle three types of input:

#### 1. Image Inference
Detect objects in a static image file.
```bash
python yolo_detect.py --model my_model/best.pt --source test_image.jpg
```

#### 2. Video Inference
Detect objects in a video file.
```bash
python yolo_detect.py --model my_model/best.pt --source test_video.mp4
```

#### 3. Webcam (Real-Time)
Run the model live using your computer's webcam.
```bash
python yolo_detect.py --model my_model/best.pt --source 0
```
*(Note: Use `0` for the default webcam, or `1` for an external camera).*

### Script Arguments
*   `--model`: Path to your trained model file (`.pt`).
*   `--source`: Input source (image path, video path, or camera index).
*   `--thresh`: (Optional) Confidence threshold. Default is 0.5 (only show detections with >50% confidence).
