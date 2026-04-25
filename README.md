# MLOps Heart Disease Prediction API

An end-to-end Machine Learning pipeline for predicting heart disease, deployed as a REST API using FastAPI and containerized with Docker.

---

## 📌 Overview

This project demonstrates a production-style MLOps workflow:

* Train a Machine Learning model
* Save model artifacts
* Serve predictions via API
* Containerize the application for consistent deployment

---

## 📊 Dataset

The dataset used for this project is sourced from Kaggle and contains medical attributes for heart disease prediction.

---

## 🛠 Tech Stack

* Python
* Scikit-learn
* FastAPI
* Uvicorn
* Docker
* Joblib

---

## ⚙️ Features

* Logistic Regression model for classification
* Model persistence using Joblib
* REST API for real-time predictions
* Dockerized for portability and scalability

---

## 📂 Project Structure

```
mlops-heart-disease-api/
│
├── data/
├── models/
├── train.py
├── predict.py
├── app.py
├── requirements.txt
├── Dockerfile
└── .gitignore
```

---

## ▶️ Run Locally

### 1. Train the model

```
python train.py
```

### 2. Run the API

```
uvicorn app:app --reload
```

### 3. Open in browser

```
http://localhost:8000/docs
```

---

## 🧪 API Usage

### Endpoint:

```
POST /predict
```

### Sample Input:

```json
{
  "features": [63,1,3,145,233,1,0,150,0,2.3,0,0,1]
}
```

### Sample Output:

```json
{
  "prediction": 1
}
```

---

## 🐳 Run with Docker

### Build Image

```
docker build -t mlops-api .
```

### Run Container

```
docker run -p 8000:8000 mlops-api
```

---

## 🎯 Key Learnings

* End-to-end ML pipeline development
* Model deployment using FastAPI
* Containerization using Docker
* Separation of training and inference workflows

---

## 📌 Future Improvements

* Add model versioning
* CI/CD pipeline integration
* Cloud deployment (AWS / GCP)
* Monitoring and logging

---

## 👨‍💻 Author

Prem Chavan
