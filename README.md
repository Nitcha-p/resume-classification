# 📄 Resume Classification Using Support Vector Machine (SVM)

## 🔍 Project Overview
This project presents an automated system to classify resumes into predefined job categories using machine learning and natural language processing (NLP). It extracts text from PDF resumes, processes it, and uses an optimized SVM model to classify resumes with high accuracy and recall.

---

## 🎯 Objectives
- Automate resume classification to support HR recruitment
- Improve performance over prior research using Random Forest
- Enhance recall in underrepresented job categories
- Build a clean, reproducible NLP pipeline

---

## 📁 Dataset
- **Source :** [Kaggle - Resume Dataset](https://www.kaggle.com/datasets/snehaanbhawal/resume-dataset/data)
- **Size :** 2,400+ resumes (PDF format)
- **Categories :** 24 job types (filtered down to 21 after validation)
- **Format :** Folder-per-category, with PDF resumes as individual files

---

## ⚙️ Methodology

### 🔹 Data Preparation
- Removed misaligned and mislabeled categories
- Validated resume-category relevance using keyword checking

### 🔹 Text Extraction
- Used [`PyMuPDF`](https://pymupdf.readthedocs.io/) to extract text from PDF resumes
- Stored results in Excel format with `Category` and `Text` columns

### 🔹 Text Preprocessing
- Expanded contractions (e.g., *can't → cannot*)
- Removed emails, phone numbers, URLs using regex
- Lowercased text, tokenized, lemmatized, and removed stopwords

### 🔹 Feature Engineering
- Applied **TF-IDF vectorization** with (1,2) n-grams and sublinear scaling
- Selected top 500 features using **Chi-Square Test**

### 🔹 Model Training
- Encoded labels using `LabelEncoder`
- Used `LinearSVC` with `GridSearchCV` for hyperparameter tuning
- Evaluated model using macro recall and class-wise metrics

---

## 📊 Results

| Metric           | Score |
|------------------|-------|
| Accuracy         | 0.82  |
| Macro Recall     | 0.82  |
| Weighted Recall  | 0.82  |

- High recall in categories like `Teacher`, `Construction`, and `HR`
- Low recall in categories like `Arts` and `Apparel`
- 6% train-test accuracy gap → minimal overfitting

---

## ✅ Key Achievements
- Developed an end-to-end machine learning pipeline
- Improved accuracy from 53% (Random Forest baseline) to 82% using SVM
- Built a clean resume classification system using only PDFs

---

## ⚠️ Limitations
- Struggles with recall in niche job categories
- Does not support image-based resumes (PDF only)
- No contextual embeddings (e.g., BERT)

---

## 🚀 Future Enhancements
- Apply **OCR (e.g., Tesseract)** to support image-based resumes
- Enhance NLP with **NER** and **POS tagging**
- Test deep learning models (e.g., **BERT**, **LSTM**)

---

## 🧰 Technologies & Libraries
- `Python` (3.10)  
- `scikit-learn`, `pandas`, `nltk`, `re`, `PyMuPDF`  
- `matplotlib`, `seaborn`, `openpyxl`

---

## 📄 License
This project is for educational and research purposes only. Dataset is publicly available on Kaggle.
