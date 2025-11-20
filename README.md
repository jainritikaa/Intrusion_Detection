
# 📘 **README.md — Network Intrusion Detection System (IDS)**

### *Using Machine Learning & Ensemble Techniques (UNSW-NB15 Dataset)*

---

## 📌 **Project Overview**

This project implements a **Network Intrusion Detection System (IDS)** using the **UNSW-NB15 cybersecurity dataset**.
The model detects whether a network connection is **Normal** or **Malicious** using:

* **Decision Tree**
* **Gaussian Naive Bayes**
* **XGBoost**
* **Voting Ensemble Classifier (Final Model)**

The ensemble classifier combines strengths of multiple models and delivers **higher accuracy and reduced false alarms** compared to individual classifiers.

A fully interactive **Streamlit web application** is included for real-time inference.

---

## 🎯 **Features**

### ✔ Data Cleaning & Preprocessing

* Handles missing values
* Label encodes categorical attributes
* Normalizes numerical features (Min-Max Scaling)

### ✔ Exploratory Data Analysis (EDA)

* Label distribution
* Attack category distribution
* Correlation heatmap

### ✔ Machine Learning Models

* Decision Tree
* Gaussian Naive Bayes
* Logistic Regression
* K-Nearest Neighbor
* Random Forest
* XGBoost
* **Voting Ensemble (Final Model)**

### ✔ Evaluation Metrics

* Accuracy
* Precision, Recall, F1-score
* Confusion Matrix
* Model comparison bar chart

### ✔ Streamlit Web App

* Upload CSV and detect attack/normal
* Run inference on sample dataset rows
* Download predictions
* Displays metrics (if ground truth available)

---

## 📂 **Project Structure**

```
📁 IDS-Project/
│
├── datasets/
│   └── UNSW_NB15.csv
│
├── models/
│   └── ensemble_ids.pkl
│
├── plots/
│   └── (generated diagrams here)
│
├── predictions/
│   └── (batch predictions saved here)
│
├── notebooks/
│   └── IDS_training.ipynb   # optional
│
├── app.py                    # Streamlit application
├── train.py                  # Main ML code
├── README.md                 # Project documentation
└── requirements.txt
```

---

## 🚀 **How to Run the Project**

### 1️⃣ **Clone the repository**

```bash
git clone https://github.com/yourusername/IDS-Ensemble.git
cd IDS-Ensemble
```

### 2️⃣ **Install dependencies**

```bash
pip install -r requirements.txt
```

### 3️⃣ **(Optional) Train the models**

If you want to retrain the ensemble model:

```bash
python train.py
```

This will generate:

```
models/ensemble_ids.pkl
```

### 4️⃣ **Run the Streamlit Web App**

```bash
streamlit run app.py
```

---

## 🧪 **Using the Streamlit App**

### **Mode 1 — Sample from Dataset**

* Pick an index from UNSW-NB15
* View raw and preprocessed features
* Run prediction

### **Mode 2 — Upload CSV**

* Upload multiple network records
* The app preprocesses automatically
* Generates predictions + summary
* Option to download results CSV

---

## 📊 **Results Summary**

The Voting Ensemble performed best among all models.

| Model                       | Accuracy |
| --------------------------- | -------- |
| Decision Tree               | ~XX%     |
| Gaussian NB                 | ~XX%     |
| Logistic Regression         | ~XX%     |
| KNN                         | ~XX%     |
| Random Forest               | ~XX%     |
| XGBoost                     | ~XX%     |
| **Voting Ensemble (Final)** | **~XX%** |

*(Fill in your actual accuracy results)*

---

## 🧱 **System Architecture**

```
Raw UNSW Dataset
        │
        ▼
Data Cleaning (missing values, symbols)
        │
        ▼
Feature Engineering (encoding + normalization)
        │
        ▼
Train Models  →  Individual Results
        │
        ▼
Voting Ensemble (final model)
        │
        ▼
Streamlit Web App → Predictions
```

---

## 🗂 **Dataset**

* Dataset: **UNSW-NB15**
* Contains 49 features including:

  * Source/destination IP
  * Protocol type
  * Service
  * Flags
  * Flow duration
  * Attack category
* Labels:

  * **0 = Normal**
  * **1 = Malicious Attack**

Dataset source: UNSW Cyber Range Lab.

---

## 💻 **Technologies Used**

* Python 3
* scikit-learn
* XGBoost
* pandas / numpy
* matplotlib / seaborn
* Streamlit
* MinMaxScaler, LabelEncoder

---

## 🛡 **Future Enhancements**

* Add deep learning models (LSTM, CNN)
* Real-time packet capture & classification
* Improve feature selection using PCA
* Deploy app using Docker + cloud hosting
* Integrate with SIEM systems

