<!-- Banner section -->
<div align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python Badge">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn Badge">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas Badge">
  <img src="https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white" alt="NumPy Badge">
  <br>
  <img src="https://img.shields.io/badge/Internship-CodSoft-blue?style=for-the-badge&logo=workplace&logoColor=white" alt="CodSoft Badge">
  <img src="https://img.shields.io/badge/Domain-Data%20Science-success?style=for-the-badge" alt="Data Science Badge">
</div>

<br>

<div align="center">
  <h1>📊 CodSoft Data Science Internship Portfolio</h1>
  <p><i>A curated collection of machine learning and data science projects built during my internship at CodSoft. This repository showcases end-to-end pipelines including data preprocessing, feature engineering, handling class imbalance, model building, and detailed evaluation.</i></p>
</div>

---

## 🗂️ Table of Contents
1. [🌟 Internship Overview](#-internship-overview)
2. [🛠️ Tech Stack & Libraries](#%EF%B8%8F-tech-stack--libraries)
3. [📂 Directory Structure](#-directory-structure)
4. [🚀 Project breakdown & Implementation](#-project-breakdown--implementation)
   - [Task 1: Titanic Survival Prediction](#task-1-titanic-survival-prediction)
   - [Task 3: Iris Flower Classification](#task-3-iris-flower-classification)
   - [Task 5: Credit Card Fraud Detection](#task-5-credit-card-fraud-detection)
5. [💡 Key Learning Outcomes](#-key-learning-outcomes)
6. [⚙️ Installation & How to Run](#%EF%B8%8F-installation--how-to-run)
7. [🎓 Conclusion](#-conclusion)

---

## 🌟 Internship Overview
During this internship, I worked on three key data science challenges starting from classic baseline models to advanced handling of highly imbalanced datasets. The focus of the tasks was:
* Clean, analyze, and preprocess real-world tabular data.
* Build robust predictive classifiers using regression and ensemble algorithms.
* Evaluate models using metrics tailored to specific business contexts (e.g. Precision, Recall, F1-Score).

---

## 🛠️ Tech Stack & Libraries
<table align="center" style="width: 100%; border: none;">
  <tr>
    <td style="width: 50%; vertical-align: top;">
      <h4>⚙️ Core Languages & Environments</h4>
      <ul>
        <li><b>Python 3.x</b> - Core scripting language</li>
        <li><b>Jupyter Notebook</b> - Interactive code development and visualization</li>
      </ul>
    </td>
    <td style="width: 50%; vertical-align: top;">
      <h4>📦 Libraries & Frameworks</h4>
      <ul>
        <li><b>Pandas & NumPy</b> - Data manipulation, cleaning, and matrix operations</li>
        <li><b>Scikit-Learn</b> - Feature scaling, model training, and metrics</li>
        <li><b>Imbalanced-Learn (SMOTE)</b> - Oversampling for minority class imbalance</li>
        <li><b>Matplotlib & Seaborn</b> - Explanatory data visualization</li>
      </ul>
    </td>
  </tr>
</table>

---

## 📂 Directory Structure
```text
md_shahid/
├── Task1_Titanic_Survival_Prediction/
│   ├── Titanic_Survival_Prediction.ipynb  # Preprocessing & Random Forest Model
│   ├── README.md                          # Task 1 Documentation
│   ├── linkedin_post.txt                  # Task 1 Summary Post
│   └── requirements.txt                   # Task 1 specific requirements
├── Task3_Iris_Flower_Classification/
│   ├── Iris_Flower_Classification.ipynb   # Exploratory Analysis & Logistic Regression
│   ├── README.md                          # Task 3 Documentation
│   ├── linkedin_post.txt                  # Task 3 Summary Post
│   └── requirements.txt                   # Task 3 specific requirements
├── Task5_Credit_Card_Fraud_Detection/
│   ├── Credit_Card_Fraud_Detection.ipynb  # SMOTE, Imbalance mitigation & Random Forest
│   ├── README.md                          # Task 5 Documentation
│   ├── linkedin_post.txt                  # Task 5 Summary Post
│   └── requirements.txt                   # Task 5 specific requirements
├── README.md                              # Main portfolio documentation (this file)
└── requirements.txt                       # Global project requirements
```

---

## 🚀 Project Breakdown & Implementation

### Task 1: Titanic Survival Prediction
<div style="background-color: #f6f8fa; padding: 15px; border-left: 5px solid #2da44e; border-radius: 4px;">
  <h4>🚢 Predictive Survival Analysis</h4>
  <p><b>Goal:</b> Build a machine learning model to predict whether a passenger survived the Titanic crash based on their socio-economic status, demographics, and ticket details.</p>
</div>

* **Dataset Characteristics:** Demographics (Age, Sex), passenger status (Pclass, Fare, Cabin, Embarked), and family details (SibSp, Parch).
* **Pipeline & Preprocessing:**
  1. **Handling Missing Data:** Filled missing `Age` and `Fare` fields using the dataset *median* values; filled missing `Embarked` entries using the *mode*.
  2. **Feature Dropping:** Discarded non-informative identifiers like `name`, `ticket`, `cabin`, `boat`, `body`, and `home.dest`.
  3. **Categorical Encoding:** Mapped `Sex` to binary numeric values (`0` for male, `1` for female) and applied One-Hot Encoding to the `Embarked` ports.
  4. **Scaling & Modeling:** Normalized features using `StandardScaler` and trained a **Random Forest Classifier** to capture complex interaction patterns.
* **Evaluation:** Scored accuracy on unseen test data and analyzed precision/recall through a standard Classification Report.

---

### Task 3: Iris Flower Classification
<div style="background-color: #f6f8fa; padding: 15px; border-left: 5px solid #0969da; border-radius: 4px;">
  <h4>🌸 Multi-class Flora Classification</h4>
  <p><b>Goal:</b> Build a classifier to identify three species of Iris flowers (Setosa, Versicolor, and Virginica) based on petal and sepal length/width measurements.</p>
</div>

* **Dataset Characteristics:** Clean, balanced dataset containing 150 instances of Iris dimensions (Sepal Length/Width, Petal Length/Width).
* **Pipeline & Preprocessing:**
  1. **Exploration:** Transformed the raw sklearn dataset into a structured Pandas DataFrame to view statistical boundaries.
  2. **Standardization:** Normalized all continuous measurements utilizing `StandardScaler` so that features contribute equally to gradients during optimizer updates.
  3. **Model Selection:** Implemented **Logistic Regression** (multiclass), a robust linear estimator for structured, linearly separable feature spaces.
* **Evaluation:** Assessed the model's accuracy, precision, and recall per flower class.

---

### Task 5: Credit Card Fraud Detection
<div style="background-color: #f6f8fa; padding: 15px; border-left: 5px solid #cf222e; border-radius: 4px;">
  <h4>💳 Fraud Identification under Extreme Imbalance</h4>
  <p><b>Goal:</b> Build a classification pipeline to flag fraudulent credit card transactions while minimizing false alarms and keeping classification sensitive to rare events.</p>
</div>

* **Dataset Characteristics:** Highly skewed dataset. Genuine transactions make up ~99.8% of the data, while fraudulent instances account for only ~0.17%. Contains numerical features obtained from PCA transformation.
* **Pipeline & Preprocessing:**
  1. **Scaling:** Normalized the `Amount` and `Time` features using `StandardScaler`.
  2. **Downsampling & SMOTE Pipeline:** 
     * For high computational efficiency on large source files, downsampled the majority class down to a manageable size.
     * Applied **SMOTE (Synthetic Minority Over-sampling Technique)** on the training data to synthetically generate minority samples and prevent the classifier from biasing towards genuine-only flags.
  3. **Modeling:** Trained a high-capacity **Random Forest Classifier** on the balanced feature space.
* **Evaluation:** Since accuracy is highly misleading here, the model was carefully optimized and verified using **Precision**, **Recall**, and **F1-Score**.

---

## 💡 Key Learning Outcomes

During this internship, I solved these three diverse classification problems and achieved several core learnings:

> [!NOTE]
> **1. Handling Real-world Missing Data & Cleaning**
> Learned when to use statistical fill-ins (medians, modes) and how feature scaling affects gradient updates and algorithm performance.

> [!TIP]
> **2. The Pitfall of Accuracy on Imbalanced Datasets**
> Discovered that a model predicting 99.8% accuracy on fraud detection could be completely useless if it fails to flag the 0.2% actual frauds. I mastered the application of **SMOTE** to balance training data and optimized the model based on **Recall** (finding all fraud) and **Precision** (avoiding false accusations).

> [!IMPORTANT]
> **3. Choosing the Right Classifier**
> Got hands-on experience comparing linear boundary classifiers (Logistic Regression) with non-linear ensemble models (Random Forest Classifier) depending on structural data distributions.

---

## ⚙️ Installation & How to Run

Follow these instructions to run the notebooks locally on your machine:

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/md_shahid.git
cd md_shahid
```

### 2. Set Up a Virtual Environment (Recommended)
On Windows:
```bash
python -m venv .venv
.venv\Scripts\activate
```
On macOS/Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Notebooks
Launch Jupyter Notebook:
```bash
jupyter notebook
```
Browse to any task folder (e.g., `Task1_Titanic_Survival_Prediction/`) and open the `.ipynb` file to run the cells sequentially.

---

<div align="center">
  <h3>🤝 Connect with me</h3>
  <p>Feel free to check out the <code>linkedin_post.txt</code> files in each task folder to see my write-ups for these projects!</p>
</div>