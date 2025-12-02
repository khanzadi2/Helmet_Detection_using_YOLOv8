🪖 Helmet Detection using YOLOv8 + Classification

Full End-to-End Computer Vision Pipeline

This repository contains a complete pipeline for detecting motorcycle riders with and without helmets using both Image Classification and Object Detection.
The project is designed for academic and practical deployment use cases, covering:

✔ Data preprocessing
✔ Classification baseline
✔ YOLOv8 object detection
✔ Robustness & error analysis
✔ Lightweight deployment
✔ Responsible AI notes
✔ Extra credit (helmet usage tracking + knowledge distillation)

📌 Dataset

Kaggle Helmet Detection Dataset by AndrewMVD:
🔗 https://www.kaggle.com/datasets/andrewmvd/helmet-detection

Annotations are provided in Pascal VOC XML and were converted to YOLO format.

📂 Repository Structure
helmet-detection-project/
│
├── README.md
│
├── classification/
│   ├── classification_training.ipynb
│   ├── confusion_matrix.png
│   └── best_classification_model.pth
│
├── detection/
│   ├── yolov8_training.ipynb
│   ├── data.yaml
│   ├── training_curves.png
│   ├── predictions/
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   │   └── ...
│   └── best_yolov8_model.pt
│
├── deployment/
│   ├── inference.py
│   ├── onnx_model.onnx
│   └── sample_video_output.mp4
│
├── error_analysis/
│   ├── robustness_results.ipynb
│   ├── failure_cases/
│   │   ├── fail1.jpg
│   │   ├── fail2.jpg
│   └── failure_modes_table.png
│
├── report/
│   ├── report.pdf
│   └── report.md
│
└── dataset/
    ├── images/
    ├── labels/
    └── raw_annotations/

🧠 A) Baseline Classification (ResNet18)
Goal:

Classify cropped rider head regions into helmet vs no-helmet.

Model:

ResNet-18 (ImageNet Pretrained)

Binary classification

20 epochs, batch=32, LR=1e-4

Metrics
Metric	Value
Accuracy	92.4%
Precision (Helmet)	91%
Recall (Helmet)	94%

Confusion matrix image included in classification/confusion_matrix.png.

Limitations

Cannot locate helmets in full images

Requires perfect crops

Cannot detect multiple riders

Not usable for CCTV streams
→ Object detection is required for real applications.

🎯 B) Object Detection (YOLOv8)
Model Details

YOLOv8s

Image size: 640

Epochs: 100

Optimizer: AdamW

Mixup & mosaic augmentations disabled for clarity

Performance:
Metric	Value
mAP@0.5	0.92
mAP@0.5:0.95	0.61
Precision	Helmet: 0.93, No-Helmet: 0.89
Recall	Helmet: 0.95, No-Helmet: 0.86

Training curves available in detection/training_curves.png.

Predictions

Over 10+ sample results stored in detection/predictions/.

🔍 C) Robustness & Error Analysis
Slicing Tests
Condition	Accuracy
Daytime	94%
Night	81%
High Occlusion	Low performance
Crowded Frames	Lower recall
Top 3 Failure Modes
Failure	Reason	Fix
Small helmets	Far camera view	Train at 768px + multi-scale
Night blur/noise	Low exposure	Gamma + blur augmentation
Occlusion	Partial visibility	Add occlusion samples

Notebook for analysis: error_analysis/robustness_results.ipynb

⚡ D) Lightweight Deployment
Run Inference
python inference.py --source image.jpg

Outputs

Bounding boxes

Class label: helmet / no-helmet

Confidence score

Hardware Speeds
Hardware	Speed
CPU (Intel i5)	~130 ms/image
GPU (T4)	~18 ms/image
Export Formats

PyTorch .pt model

ONNX for CPU acceleration

Post-processing

NMS threshold: 0.5

Confidence threshold: 0.35

Optimized for CCTV noise + small objects.

🔐 E) Responsible AI Notes

Helmet detection involves analyzing public CCTV footage, which raises privacy and fairness concerns. Models may misclassify cultural head coverings or perform poorly at night, potentially causing unjust penalties. Bias occurs if training data contains mostly daytime images or lacks diversity in clothing or helmet types.

To reduce risks:

Avoid storing personal data

Disable face recognition

Include diverse training images

Allow human review before enforcement

Document model assumptions and limitations

Comply with local data protection laws

A full write-up is included in report/report.pdf.

🏆 Extra Credit (Included)
✔ Helmet Usage Rate Tracking

Frame-wise detection

Rolling average of helmet usage

Graph visualization

✔ Knowledge Distillation

Teacher: YOLOv8m

Student: YOLOv8n

Student is 60% smaller and faster

Minor accuracy loss

🚀 How to Run This Project
1. Clone the repo
git clone https://github.com/your-username/helmet-detection-project.git
cd helmet-detection-project

2. Install dependencies
pip install -r requirements.txt

3. Download Kaggle dataset

Place dataset under:

dataset/raw_annotations/

4. Convert to YOLO format

Already handled in the notebook yolov8_training.ipynb.

5. Train Models

Classification: classification_training.ipynb

YOLOv8: yolov8_training.ipynb

6. Run Inference
python deployment/inference.py --source image.jpg

👩‍💻 Author

Khanzadi
Helmet Detection — Deep Learning Computer Vision Project
Feel free to fork, star ⭐, and improve! 
⭐ If this repo helped you, please give it a star! ⭐
