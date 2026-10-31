# Explainable AI for Diabetic Retinopathy Screening in Rural India
An AI-powered system designed to support early diabetic retinopathy screening in rural India through retinal image analysis, visual explanations, and specialist review.
## 1. Project Overview
Diabetic Retinopathy (DR) is an eye condition caused by diabetes. Early detection can help prevent serious vision problems.
This project uses deep learning to classify retinal images into five DR severity levels. It also identifies cases that may need specialist attention and uses Grad-CAM to explain model predictions.
The long-term goal is to connect patients, rural screening workers, and ophthalmologists through a single platform.

## 2. Getting Started
### Step 1: Clone the repository
```bash
git clone https://github.com/anyasahi-yowaimo/Aptos_DR_Screening.git
cd Aptos_DR_Screening
```
### Step 2: Download the trained model
```bash
git lfs install
git lfs pull
```
The model is stored at:
```text
checkpoints/best_model.pt
```
### Step 3: Create and activate a virtual environment
```bash
python -m venv venv
venv\Scripts\activate
```
### Step 4: Install dependencies
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```
### Step 5: Run the application
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

**Note:** The APTOS 2019 dataset must be prepared separately before training or evaluation.

## 3. Current Implementation
The current prototype uses:

| Component | Technology |
|---|---|
| Language | Python |
| Deep Learning | PyTorch |
| Model | EfficientNet-B0 |
| Image Processing | OpenCV |
| Augmentation | Albumentations |
| Evaluation | scikit-learn |
| Explainable AI | Grad-CAM |
| Interface | Streamlit |
| Dataset | APTOS 2019 |

### Application Features
- Data Setup
- Model Training
- Fine-tuning
- Model Evaluation
- Retinal Image Prediction
- Grad-CAM Visualization
## 4. DR Classification
The model predicts one of five severity levels:
| Class | Severity |
|---|---|
| 0 | No DR |
| 1 | Mild NPDR |
| 2 | Moderate NPDR |
| 3 | Severe NPDR |
| 4 | Proliferative DR |

For screening, the classes are grouped as:
- **Non-referable:** Classes 0, 1,2
- **Referable:** Classes 3 and 4

## 5. Model Performance
**Dataset:** APTOS 2019  
### Evaluation Results
| Metric | Result |
|---|---:|
| Five-class Accuracy | 81.55% |
| Macro F1-Score | 0.702 |
| Macro Precision | 0.696 |
| Macro Recall | 0.615 |
| Referable ROC-AUC | 0.9539 |
| Referable Sensitivity | 94.17% |
| Referable Specificity | 89.02% |

*Referable sensitivity and specificity measure referable-case screening performance, not five-class accuracy.*

## 6. Explainable AI – Grad-CAM
Grad-CAM provides visual explanations of model predictions.
The application displays:
- Original retinal image
- Grad-CAM heatmap
- Heatmap overlay
These visualizations highlight image regions that influence the model's prediction. They are intended to support interpretation, not replace specialist assessment.

## 7. Proposed Complete System
The future platform will have three portals:
**Patient Portal**
- Patient registration
- Screening appointments
- Screening results and referral status

**Screening Worker Portal**
- Patient registration
- Retinal image capture and submission
- Screening status and workload tracking

**Doctor Portal**
- Patient records and retinal images
- AI predictions and Grad-CAM explanations
- Case review, referrals and follow-ups

Doctors will be able to review and confirm or override AI-assisted results.

## 8. Village-Level Resource Management
The proposed platform will help plan rural screening resources using:
- Village and estimated diabetic population
- Expected screening demand
- Available workers and fundus cameras
- Screening-centre capacity
- Doctor availability and review workload

It aims to identify resource shortages, waiting times, and operational bottlenecks.

## 9. Referral and Specialist Discovery
The planned referral system will help:
- Identify cases requiring specialist review.
- Find verified nearby ophthalmologists and eye-care facilities.
- Support referrals and follow-up tracking.

## 10. Proposed Technology Architecture
| Component | Proposed Technology |
|---|---|
| Frontend | React, Next.js, TypeScript |
| Backend | FastAPI |
| Database | PostgreSQL |
| Image Storage | MinIO / S3-compatible storage |
| Medical AI | PyTorch, MONAI |
| Image Processing | OpenCV |
| Explainability | Grad-CAM |
| Simulation | MATLAB, Simulink |
| Security | OAuth 2.0, OIDC, RBAC |
| Deployment | Docker, Kubernetes |
| Monitoring | Prometheus, Grafana |

## 11. MATLAB / Simulink Integration
MATLAB/Simulink is planned to simulate the rural screening workflow, including:

- Patient arrivals
- Worker and camera utilization
- Image transfer and AI processing
- Doctor-review queues
- Waiting times and resource requirements

**AI model:** Analyzes retinal images.

**Simulink:** Evaluates whether the screening infrastructure can handle the expected workload.

## 12. Future Development
- Lesion detection and segmentation
- Doctor review and referral workflow
- Nearby specialist discovery
- Docker and Kubernetes deployment
- Prometheus and Grafana monitoring
- External validation using IDRiD and Messidor-2
- Clinical and human-in-the-loop evaluation

## 13. Current Limitations
- Currently trained and evaluated on APTOS 2019.
- Performance on external datasets is not yet established.
- Rural image variations require further validation.
- Clinical validation is still required.
- The complete portal, referral and resource-management system is on going process.
This is an AI-assisted screening research prototype, not a replacement for professional medical diagnosis.
## 14. Repository Structure
```text
Aptos_DR_Screening/
│
├── app.py
├── checkpoints/
│   └── best_model.pt
├── requirements.txt
├── model/
├── dataset/
└── README.md
```

## 15. Project Information
- **Project:** Explainable AI for Diabetic Retinopathy Screening in Rural India
- **Hackathon:** Smart India Hackathon 2026
- **Problem Statement:** SIH26038
- **Category:** Healthcare / Medical AI / HealthTech
- **Current Stack:** Python, PyTorch, EfficientNet-B0, Grad-CAM, Streamlit
- **Repository:** [Aptos_DR_Screening](https://github.com/anyasahi-yowaimo/Aptos_DR_Screening)

*Building accessible and explainable AI-assisted retinal screening for rural India.*
