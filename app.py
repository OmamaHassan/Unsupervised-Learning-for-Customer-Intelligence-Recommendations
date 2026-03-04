import streamlit as st
import pandas as pd
import numpy as np
from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# -------------------------------
# Load preprocessed data
# -------------------------------
@st.cache_data
def load_data():
    rfm = pd.read_csv("rfm_processed.csv")
    df_encoded = pd.read_csv("df_encoded.csv")
    user_item_sparse = load_npz("user_item_sparse.npz")
    with open("product_mapping.pkl", "rb") as f:
        product_mapping = pickle.load(f)
    return rfm, df_encoded, user_item_sparse, product_mapping

rfm, df_encoded, user_item_sparse, product_mapping = load_data()


# -------------------------------
# Fit k-NN model for recommendations
# -------------------------------
@st.cache_resource
def fit_knn_model(_sparse_matrix, n_neighbors=6):
    model = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=n_neighbors)
    model.fit(_sparse_matrix)
    return model

model_knn = fit_knn_model(user_item_sparse)


# -------------------------------
# Recommendation function
# -------------------------------
def recommend_products(user_index, user_item_matrix, model, product_mapping, n_recommendations=5):
    distances, indices = model.kneighbors(user_item_matrix[user_index], n_neighbors=6)
    similar_users = indices.flatten()[1:]  # skip the user itself
    recommended_products = set()
    for u in similar_users:
        purchased = user_item_matrix[u].nonzero()[1]
        recommended_products.update(purchased)
    already_purchased = set(user_item_matrix[user_index].nonzero()[1])
    recommended_products -= already_purchased
    return [product_mapping[p] for p in recommended_products][:n_recommendations]


# -------------------------------
# Streamlit UI
# -------------------------------
st.title("E-Commerce Customer Analysis & Recommendations")

# Customer selection
customer_list = rfm['customer_id'].tolist()
selected_customer = st.selectbox("Select a Customer ID", customer_list)

# Get customer RFM & PCA info
customer_rfm = rfm[rfm['customer_id'] == selected_customer]

# Get encoded customer index for recommendations
customer_idx = df_encoded[df_encoded['customer_unique_id'] == selected_customer]['customer_id_encoded'].iloc[0]



# ------------------------------
# Tabs for main app + analysis
# ------------------------------
tab1, tab2, tab3 = st.tabs(["Customer Info & RFM", "Recommendations", "Analysis Visuals"])

# Tab 1: Customer Info & RFM
with tab1:
    st.subheader("Customer RFM Metrics")
    
    # Show metrics in columns
    col1, col2, col3 = st.columns(3)
    col1.metric("Recency (Days)", customer_rfm['Recency'].values[0])
    col2.metric("Frequency (Orders)", customer_rfm['Frequency'].values[0])
    col3.metric("Monetary (Currency)", customer_rfm['Monetary'].values[0])
    
    # Display full RFM row with units
    customer_rfm_display = customer_rfm.rename(columns={
        'Recency':'Recency',
        'Frequency':'Frequency',
        'Monetary':'Monetary',
        'Cluster':'Cluster',
        'Anomaly_KDE':'Anomaly Flag'
    })
    st.dataframe(customer_rfm_display)


# Tab 2: Recommendations
# Show recommendations
with tab2:
    st.subheader("Top Product Recommendations")
    recommended = recommend_products(customer_idx, user_item_sparse, model_knn, product_mapping, n_recommendations=5)
    if recommended == []:
        st.write("No recommendations!")
    else:
        recommended_df = pd.DataFrame(recommended, columns=["Recommended Product IDs"])
        st.table(recommended_df)


# Tab 3: Analysis Visuals
with tab3:
    
    # PCA scatter plot for visualization
    st.subheader("RFM PCA Scatter Plot")
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x='PC1', y='PC2',
        hue='Cluster',
        data=rfm,
        palette='Set1',
        alpha=0.6,
        ax=ax
    )
    sns.scatterplot(
        x=customer_rfm['PC1'],
        y=customer_rfm['PC2'],
        color='black',
        s=100,
        label='Selected Customer',
        ax=ax
    )
    ax.set_title("PCA of RFM Features with Clusters")
    st.pyplot(fig)

    #-----
    
    st.subheader("KDE Anomaly Detection (PCA Projection)")
    fig2, ax2 = plt.subplots(figsize=(8,6))
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Anomaly_KDE', 
        data=rfm, 
        palette={0:'blue',1:'red'},
        alpha=0.6,
        ax=ax2
    )
    ax2.set_title("KDE Anomalies: Blue=Normal, Red=Anomalies")
    st.pyplot(fig2)

    #-----

    st.subheader("Cluster Distribution (Bar Plot)")

    fig, ax = plt.subplots(figsize=(6,4))
    sns.countplot(x='Cluster', data=rfm, palette='Set1', ax=ax)
    ax.set_title("Number of Customers per Cluster")

    # Add numbers above each bar
    for p in ax.patches:
        height = p.get_height()
        ax.text(
            x=p.get_x() + p.get_width()/2,  # x-coordinate (center of bar)
            y=height + 50,                  # y-coordinate slightly above bar
            s=int(height),                  # label = bar height
            ha='center'                     # horizontal alignment
        )

    st.pyplot(fig)