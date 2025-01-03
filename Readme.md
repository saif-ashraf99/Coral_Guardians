# Coral Guardians: A Generative AI-Based Underwater Monitoring System

**Coral Guardians** is a demonstration project that uses Generative Adversarial Networks (GANs) to create synthetic underwater images—helping to augment datasets for coral reef detection. The project also leverages [YOLOv5](https://github.com/ultralytics/yolov5) for object detection and **Streamlit** for a simple web-based dashboard.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Project Structure](#project-structure)
4. [Installation &amp; Setup](#installation--setup)
5. [Data Preparation](#data-preparation)
6. [Training the Generative Model (DCGAN)](#training-the-generative-model-dcgan)
7. [Object Detection (YOLOv5)](#object-detection-yolov5)
8. [Inference](#inference)
9. [Dashboard (Streamlit)](#dashboard-streamlit)
10. [Next Steps &amp; Extensions](#next-steps--extensions)
11. [License](#license)
---
## 1. Project Overview

The **Coral Guardians** project aims to:

* **Monitor coral reefs** by detecting coral species, fish, debris, and signs of coral bleaching.
* **Enhance training datasets** with synthetic images generated by a DCGAN, thus improving detection accuracy under varied underwater conditions (lighting, turbidity, etc.).
* **Provide a streamlined interface** (via Streamlit) for quick image-based inference and visualization of detection results.

 **Disclaimer** : The code is an example or “starter” framework. It is *not* production-ready. Further customization, data curation, and model optimization are needed for real-world deployment.

---

## 2. Features

* **Generative Model (DCGAN)** for synthetic underwater image creation.
* **Object Detection** using [YOLOv5](https://github.com/ultralytics/yolov5).
* **Simple Data Loading & Annotation Tools** (sample scripts for YOLO-style annotation).
* **Streamlit Dashboard** to visualize detections on uploaded images.
* **Modular Project Structure** to facilitate integration of additional models and data sources.

---

## 3. Project Structure

A suggested folder layout:

```
coral_guardians/
│
├── data/
│   ├── real_images/
│   │   ├── class1/ (if you want to use ImageFolder)
│   │   ├── class2/
│   │   └── ...
│   ├── annotations/
│   │   └── ...        # YOLO-format bounding box labels
│   └── synthetic_images/
│       └── ...        # Generated images from DCGAN
│
├── generative/
│   ├── dcgan_train.py
│   └── dcgan_models.py
│
├── detection/
│   ├── yolov5/        # Cloned YOLOv5 repository
│   ├── train_detector.py
│   └── inference.py
│
├── dashboard/
│   └── streamlit_app.py
│
└── utils/
    ├── dataset_utils.py
    ├── annotation_tools.py
    └── ...
```

---

## 4. Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/saif-ashraf99/coral_guardians.git
   cd coral_guardians
   ```
2. **Set up a Python environment** (conda or virtualenv recommended)
   ```bash
   conda create --name coral_env python=3.11 -y
   conda activate coral_env
   ```
3. **Install Dependencies**
   * Install PyTorch (with CUDA if you have a GPU) following [PyTorch Installation Instructions](https://pytorch.org/).
   * Inside the `coral_guardians` folder:
     ```bash
     pip install opencv-python
     pip install streamlit
     pip install torchvision
     pip install tqdm
     ```
   * Clone YOLOv5:
     ```bash
     cd detection
     git clone https://github.com/ultralytics/yolov5.git
     cd yolov5
     pip install -r requirements.txt
     cd ../..
     ```
4. **Verify Installation**
   * Ensure you can import packages without errors:
     ```bash
     python -c "import torch; import cv2; import streamlit; print('Setup complete!')"
     ```

---

## 5. Data Preparation

1. **Collect real underwater images** of corals, fish, and other marine life.
2. **Organize images** under `./data/real_images/`.
3. **Annotate** bounding boxes for YOLOv5:
   * Each image needs a corresponding `.txt` file listing bounding box coordinates in YOLO format.
   * Alternatively, use a tool like [CVAT](https://github.com/opencv/cvat), [LabelStudio](https://github.com/heartexlabs/label-studio), or [LabelMe](https://github.com/wkentaro/labelme).
4. **(Optional) Separate** your data into `train/`, `val/`, and `test/` folders if you want a more structured approach.
5. **Adjust** `data.yaml` (in the `detection/` folder) to point to these folders, and list the class names (e.g. `coral, fish, debris`).

---

## 6. Training the Generative Model (DCGAN)

1. **Move to the generative folder** :

```bash
   cd generative
```

1. **Run `dcgan_train.py`** :

```bash
   python dcgan_train.py \
       --real_images_folder ../data/real_images \
       --output_folder ../data/synthetic_images \
       --num_epochs 50 \
       --batch_size 32 \
       --device cuda
```

* This script will:
  * Load real images.
  * Train a simple DCGAN.
  * Generate synthetic images each epoch.
  * Save `dcgan_generator.pth` and `dcgan_discriminator.pth` in `output_folder`.

1. **Use Synthetic Images** :

* Check the generated images under `./data/synthetic_images/`.
* Optionally **add** (or  **merge** ) these synthetic images and corresponding annotations (if any) into your training set for YOLOv5.

---

## 7. Object Detection (YOLOv5)

### 7.1 YOLOv5 Data Configuration

* **`data.yaml`** (example):
  ```yaml
  train: ../data/train/images
  val: ../data/val/images
  test: ../data/test/images

  nc: 3  # Number of classes
  names: ['coral', 'fish', 'debris']
  ```

### 7.2 Training YOLOv5

From the `detection` folder:

```bash
python train_detector.py \
    --data_yaml data.yaml \
    --imgsz 640 \
    --batch_size 8 \
    --epochs 50 \
    --device 0 \
    --name coral_model
```

 **What happens** :

* `train_detector.py` calls the official YOLOv5 `train.py` script with your configuration.
* Model checkpoints (weights) are saved under `yolov5/runs/train/coral_model/weights/`.

---

## 8. Inference

Once training completes, use `inference.py` to run detection on a test image:

```bash
cd detection
python inference.py
```

* Ensure you’ve set the correct `weights` path in `inference.py` (e.g., `./yolov5/runs/train/coral_model/weights/best.pt`).
* Replace `test_image.jpg` with an actual underwater image path.

 **Output** : A list of detections (bounding boxes, confidence, and class).

---

## 9. Dashboard (Streamlit)

You can run a simple GUI for inference:

```bash
cd dashboard
streamlit run streamlit_app.py
```

1. The browser will open (or visit [http://localhost:8501](http://localhost:8501) if it doesn’t auto-open).
2. Upload an underwater image (JPG/PNG).
3. The dashboard will run inference, draw bounding boxes, and display the results.

 **Note** : By default, `streamlit_app.py` loads weights from `./detection/yolov5/runs/train/coral_model/weights/best.pt`. Adjust that path as needed.

---

## 10. Next Steps & Extensions

* **Advanced Generative Models** : Experiment with StyleGAN, diffusion models, or other data augmentation methods for more realistic synthetic images.
* **Semantic Segmentation** : For detailed coral coverage or bleaching severity analysis, consider U-Net or Mask R-CNN.
* **Edge Deployment** : Optimize YOLOv5 (or other networks) to run on embedded devices or underwater drones.
* **3D Reconstruction** : Use photogrammetry to build 3D maps of reefs and track health over time.
* **Real-Time Monitoring** : Set up live video streaming from underwater cameras, feeding into the detection pipeline for immediate alerts on debris or bleaching events.
