# Potato Disease Classification – End-to-End Machine Learning Pipeline  
### African Leadership University — Machine Learning Pipeline Summative Assignment  
**Author:** Edine Noella Mugisha  
**Year:** 2025  

---

## Project Overview  
This project implements a complete **Machine Learning Pipeline** for classifying potato leaf diseases using image data.  
It demonstrates:

- Data acquisition & preprocessing  
- Model training (CNN & Transfer Learning)  
- Model evaluation & visualization  
- API development (Flask)  
- Model prediction (single & batch)  
- Retraining with user-uploaded data  
- Monitoring (uptime, prediction stats, CPU usage, memory usage)  
- Web dashboard UI  
- Deployment to a cloud platform  
- Load testing with Locust  

This project satisfies **all requirements** in the ALU Machine Learning Pipeline Summative.

---

## Live Demo (Hosted Application)
**Live App URL:** *Add your Render URL here*  

---

## 🎥 Video Demonstration  
**YouTube Demo:** *Add your YouTube link here*  

The video demonstrates:  
✔ UI walkthrough  
✔ Predictions (single & batch)  
✔ Retraining trigger  
✔ Visualization page  
✔ Dashboard (uptime, system stats, prediction stats)  
✔ API demonstration  
✔ Flood test results  

---

# Project Structure

potato_disease_summative_MLOP/
│
├── README.md
│
├── notebook/
│ ├── potato_disease_classification.ipynb
| └──results
│
├── src/
│ ├── api.py
│ ├── prediction.py
│ ├── train.py
│ ├── preprocessing.py
│ ├── model.py
│ └── init.py
│
├── templates/
│ ├── index.html
│ ├── dashboard.html
│ └── visualizations.html
│
├── static/
│ ├── css/style.css
│ └── js/main.js
│
├── models/
│ └── potato_mobilenetv2_YYYYMMDD_HHMMSS/
│ ├── potato_mobilenetv2.h5
│ ├── class_names.json
│ └── training_history.json
│
├── data/
│ ├── potato/
│ └── upload/ # User-uploaded images for retraining
│
|
├── logs/
│ └── training.log
│
├── tests/
│ └── locustfile.py
│
└── requirements.txt


---

# Machine Learning Pipeline Summary

## 1. Data Acquisition  
- Dataset: **Potato Leaf Dataset (subset of PlantVillage)**  
- Classes:  
  - *Potato___Early_blight*  
  - *Potato___Late_blight*  
  - *Potato___healthy*  
- Loaded into `data/train/` & `data/test/`.

---

## 2. Data Processing  
Performed in `DataPreprocessor` (in `preprocessing.py`):

- Resizing (256×256)
- Normalization
- Data augmentation  
- Train/Validation split  
- Batch generators  

---

## 3. Model Creation  
Implemented in `model.py`:

- **Custom CNN model**  
- **Transfer Learning (MobileNetV2)**

Key layers include:
- Conv2D blocks  
- Batch normalization  
- Dropout regularization  
- GlobalAveragePooling2D  
- Dense classification head  

---

## 4. Model Training  
Training pipeline is implemented in `train.py`.

✔ Early stopping  
✔ Learning rate scheduling  
✔ Checkpoint saving (`.h5`)  
✔ Tensorboard logging  
✔ Training history JSON export  

Metrics Tracked:
- Accuracy  
- Precision  
- Recall  
- Loss  

---

## 5. Model Evaluation & Visualizations  
Evaluation includes:

- Confusion matrix  
- Classification report  
- ROC curves  
- Accuracy/Loss curves  

All evaluation steps are shown in the Jupyter Notebook.

---

## 6. Model Deployment & Prediction API  
`src/api.py` exposes:

### Prediction Endpoints  
| Endpoint | Description |
|---------|-------------|
| `POST /api/predict` | Single image prediction |
| `POST /api/predict/batch` | Batch prediction for multiple images |

### Response includes:
- predicted class  
- confidence score  
- class-wise confidence  
- inference time  
- interpretation (description + recommendation)

---

## 7. Retraining Pipeline  
Users can:

✔ Upload new images (bulk)  
✔ Assign them to a class  
✔ Trigger retraining (`POST /api/train/retrain`)  
✔ Monitor progress on the dashboard  

Retraining happens on a background thread.

---

## 8. Monitoring Dashboard  
`dashboard.html` connects to:

| Endpoint | Use |
|---------|------|
| `/api/monitor/uptime` | Shows model uptime |
| `/api/monitor/stats` | Shows CPU, RAM, predictions made |
| `/api/train/status` | Shows training/retraining progress |

UI features:
- Training progress bar  
- System statistics  
- Inference performance statistics  
- Upload + retrain controls  

---

## 9. Visualizations Page  
`visualizations.html` displays:

- Class distribution  
- Disease impact radar chart  
- Learning curves (simulated or real)  
- Key insights explaining dataset patterns  

---

# 🏃‍♂️ How to Run Locally

## 1. Clone the repo

```bash
git clone https://github.com/edine-noella/potato_disease_summative_MLOP
cd potato_disease_summative_MLOP
```

## 2. Create & activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate     # Mac/Linux
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```
## 4. Run the Flask API

```bash
python -m src.api
```
Your app will be available at:

http://127.0.0.1:5000

## 5. Run Jupyter Notebook (optional)

```bash
jupyter notebook notebook/potato_disease_classification.ipynb
```
