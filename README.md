# E-Commerce Customer Segmentation & Recommendation System

A complete end-to-end data science project that performs **RFM-based customer segmentation**, **K-Means clustering**, **KDE anomaly detection**, **PCA visualization**, and **collaborative filtering product recommendations**, deployed through an interactive **Streamlit web application**.

## Dataset Source
Kaggle - Brazilian E-Commerce Public Dataset by Olist


## Project Overview

This project analyzes e-commerce transaction data to:

- Segment customers using **RFM Analysis**
- Identify customer groups using **K-Means Clustering**
- Detect unusual customers using **Kernel Density Estimation (KDE)**
- Visualize customer patterns using **Principal Component Analysis (PCA)**
- Generate personalized product recommendations using **User-Based Collaborative Filtering**
- Provide an interactive dashboard via **Streamlit**


## Methodology

### 1️. RFM Analysis

RFM metrics were calculated per customer:

- **Recency (Days)** → Days since last purchase  
- **Frequency (Orders)** → Total number of purchases  
- **Monetary (Currency)** → Total amount spent  

These features summarize customer engagement and value.


### 2️. Feature Scaling

Scaling was applied to normalize feature ranges before clustering and anomaly detection.

**Why scaling?**
- Ensures no feature dominates due to larger numerical magnitude.
- Improves clustering performance.


### 3️. K-Means Clustering

Customers were segmented using **K-Means** based on scaled RFM features.

Outputs:
- `Cluster` column added to RFM dataset
- Customer groups representing different behavioral patterns


### 4️. KDE Anomaly Detection

Kernel Density Estimation was used to identify customers with unusual purchasing behavior.

Outputs:
- `Anomaly_KDE` column (0 = normal, 1 = anomaly)

**Why outliers were not removed?**
Outliers may represent high-value or unique customers, which are important for business strategy.


### 5️. PCA (Principal Component Analysis)

PCA was applied to:

- Reduce dimensionality (3D → 2D)
- Visualize clusters and anomalies clearly

Outputs:
- `PC1`
- `PC2`

PCA is used only for visualization, not for clustering.


### 6️. Recommendation System

A **User-Based Collaborative Filtering** approach was implemented:

- User-item interaction matrix built using sparse matrix
- Cosine similarity used via k-Nearest Neighbors
- Recommends products based on similar users' purchases


### 7. Streamlit Web Application

The final app includes:

### Tabs

1. **Customer Info & RFM**
   - RFM metrics displayed with units
   - Cluster and anomaly status
   - PCA coordinates

2. **Recommendations**
   - Top N product recommendations

3. **Analysis Visuals**
   - K-Means cluster visualization
   - KDE anomaly detection visualization
   - Cluster distribution bar chart with labels


## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Seaborn
- Matplotlib
- Plotly (optional interactive visuals)
- Streamlit


## Project Structure

```
├── data/
├── Analysis.ipynb # Detailed analysis, training and evaluation
├── streamlit_app.py 
├── rfm_processed.csv
├── df_encoded.csv
├── user_item_sparse.npz
├── README.md

```


