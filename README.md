# 🚀 AI & ML Internship — Task 12
# 🧠 KMeans Customer Segmentation

---

## 📌 Objective
Perform **Customer Segmentation using KMeans Clustering** to group mall customers based on:

- 💸 Annual Income
- 🛍 Spending Score

This helps businesses understand customer behavior and apply **targeted marketing strategies**.

---

## 🛠 Tools & Libraries Used
- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (KMeans, StandardScaler)

---

## 📂 Dataset
**Mall Customer Segmentation Dataset (Kaggle)**

### Features Used:
- Annual Income (k$)
- Spending Score (1–100)

### Removed:
- CustomerID (not useful for clustering)

---

# ⚙️ Step-by-Step Workflow

---

## ✅ Step 1 — Load Dataset
```python
df = pd.read_csv("Mall_Customers.csv")
```

✔ Checked shape, info, describe  
✔ Dropped unnecessary columns  

---

## ✅ Step 2 — Feature Scaling

KMeans uses **distance**, so scaling is mandatory.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

✔ Mean ≈ 0  
✔ Std ≈ 1  

---

## ✅ Step 3 — Elbow Method (Find Optimal K)

```python
inertia = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
```

📉 **Optimal K found = 5**

---

## ✅ Step 4 — Train KMeans Model

```python
kmeans = KMeans(n_clusters=5, random_state=42)
labels = kmeans.fit_predict(X_scaled)

df["Cluster"] = labels
```

✔ Cluster labels added to dataset

---

## ✅ Step 5 — Visualize Clusters

```python
plt.scatter(
    df["Annual Income (k$)"],
    df["Spending Score (1-100)"],
    c=df["Cluster"]
)
```

✔ Clear separation of customer groups  
✔ Centroids plotted  

---

## ✅ Step 6 — Cluster Interpretation

```python
df.groupby("Cluster").mean(numeric_only=True)
```

### Customer Segments

| Cluster | Type | Description |
|--------|----------------------|------------------------------|
| 0 | Budget Customers | Low income, low spending |
| 1 | Conservative | High income, low spending |
| 2 | Average | Medium income, medium spending |
| 3 | Impulsive | Low income, high spending |
| 4 | Premium / VIP ⭐ | High income, high spending |

---

## ✅ Step 7 — Export Segmented Dataset

```python
df.to_csv("segmented_customers.csv", index=False)
```

✔ Final segmented CSV generated

---

# 📊 Results

### 🎯 Elbow Plot
Helps determine optimal number of clusters (K = 5)

### 🎯 Cluster Visualization
Shows 5 clearly separated customer groups

### 🎯 Business Insight
Businesses can:
- Offer discounts to budget customers
- Use ads/promotions for conservative customers
- Maintain regular marketing for average customers
- Target impulsive buyers with trendy products
- Provide loyalty programs for VIP customers

---

# 📁 Project Structure

```
Task_12_KMeans_Customer_Segmentation/
│
├── Task_12_KMeans.ipynb
├── Mall_Customers.csv
├── segmented_customers.csv
├── README.md
```

---

# 🧠 Key Concepts Learned

- Unsupervised Learning
- KMeans Clustering
- Feature Scaling
- Elbow Method
- Data Visualization
- Business Interpretation of ML

---

# 💡 Interview Questions

### What is clustering?
Grouping similar data points without labels.

### Why scaling matters in KMeans?
Because KMeans uses distance; large values dominate otherwise.

### What is inertia?
Sum of squared distances of points to their nearest centroid.

### What is Elbow Method?
Technique to find optimal number of clusters.

### Limitations of KMeans?
- Need to choose K manually
- Sensitive to outliers
- Assumes spherical clusters

---

# ✅ Final Outcome

✔ Customers segmented into 5 groups  
✔ Visual insights generated  
✔ Business strategies derived  
✔ Real-world marketing use case demonstrated  

---

# 🎉 Task Status
✅ Completed Successfully

---
