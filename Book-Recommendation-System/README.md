# <center> **PROJECT: Book Recommendation Service**  

End-to-end hybrid book recommendation system combining classical methods with a modern neural **Wide & Deep** architecture.

---

### **Project Goal**

Develop a high-quality hybrid recommender that integrates traditional algorithms with neural networks to improve personalization and ranking metrics.

---

### **Dataset**

Book ratings dataset with user-book interactions and book metadata (titles, tags, authors).

---

### **Implemented Approaches**

- **Classical Models**: Popularity, Content-based (TF-IDF + Cosine), Item-Based CF, Matrix Factorization (SVD)
- **Hybrid System**: Candidate generation + weighted ensemble
- **Neural Model**: **Wide & Deep** architecture (memorization + generalization)

---

### **Technologies Used**

**Core Stack:**
- **Data Processing**: `pandas`, `numpy`
- **Classical RecSys**: `scikit-learn`, `Surprise`, `scipy`
- **Neural Networks**: `PyTorch` (Wide & Deep model)
- **Feature Engineering**: `TfidfVectorizer`, `cosine_similarity`, `LabelEncoder`, `StandardScaler`
- **Evaluation**: `sklearn.metrics`, `nDCG`, `Precision@K`, `Recall@K`
- **Visualization**: `matplotlib`, `seaborn`, `plotly.express`

**Additional**: `torch.nn`, `torch.optim`, GPU support

---

### **Evaluation Results**

| Model                    | Precision@5 | Recall@5  | nDCG@5   | MAP@5    | Users Evaluated |
|--------------------------|-------------|-----------|----------|----------|-----------------|
| Popularity               | 0.000362    | 0.000611  | 0.001056 | 0.000370 | 42,568          |
| Content-based            | 0.000000    | 0.000000  | 0.000000 | 0.000000 | 28              |
| Item-Based CF            | 0.070345    | 0.214114  | 0.225024 | 0.132960 | 145             |
| Matrix Factorization     | 0.001587    | 0.001235  | 0.004723 | 0.000849 | 756             |
| **Wide & Deep**          | **0.000456**| **0.000867** | **0.001193** | **0.000473** | **42,568** |

---

### **Project Stages**

1. Exploratory Data Analysis
2. Data Preprocessing & Feature Engineering
3. Classical Models Implementation
4. Hybrid Recommendation System
5. Wide & Deep Neural Model Development
6. Integration & Final Evaluation
7. Conclusions

---

### **Project Structure**

- `notebooks/` — main Jupyter notebooks
- `src/` — source code and model implementations
- `data/` — datasets
- `models/` — saved models
- `figures/` — visualizations and metric plots
- `requirements.txt`

---

### **How to run**

```bash
cd Book-Recommendation-System

pip install -r requirements.txt

jupyter notebook "PROJECT - Book Recommendation System - Hybrid + Wide&Deep.ipynb"