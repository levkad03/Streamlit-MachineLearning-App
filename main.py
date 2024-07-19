import pandas as pd
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Page Layout

st.set_page_config(page_title="First Streamlit Project", layout="wide")


st.write("""
         # The Machine Learning App
         In this implementation, the *RandomForestRegressor()* function is used in this 
         app for build a regression model using the **Random Forest** algorithm.

        Try adjusting the hyperparameters!
         """)

with st.sidebar.header("1. Upload your CSV data"):
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    st.sidebar.markdown("""
    [Example CSV input file](https://raw.githubusercontent.com/dataprofessor/data/master/delaney_solubility_with_descriptors.csv)
    """)

with st.sidebar.header("2.Set Parameters"):
    split_size = st.sidebar.slider(
        "Data split ratio (% for Trainig set)", 10, 90, 80, 5
    )

with st.sidebar.subheader("2.1 Learning Parameters"):
    parameter_n_estimators = st.sidebar.slider(
        "Number of estimators", 0, 1000, 100, 100
    )
    parameters_max_features = st.sidebar.selectbox(
        "Max features", options=["auto", "sqrt", "log2"]
    )
    parameter_min_samples_split = st.sidebar.slider(
        "Minimum number of samples required to split an internal node",
        1,
        10,
        2,
        1,
    )
    parameter_min_samples_leaf = st.sidebar.slider(
        "Minimum number of samples required to be at a leaf node",
        1,
        10,
        2,
        1,
    )
    with st.sidebar.subheader("2.2. General Parameters"):
        parameter_random_state = st.sidebar.slider(
            "Seed number (random_state)", 0, 1000, 42, 1
        )
        parameter_criterion = st.sidebar.selectbox(
            "Performance measure (criterion)", options=["mse", "mae"]
        )
        parameter_bootstrap = st.sidebar.selectbox(
            "Bootstrap samples when building trees (bootstrap)", options=[True, False]
        )
        parameter_oob_score = st.sidebar.selectbox(
            "Whether to use out-of-bag samples to estimate the R^2 on unseen data",
            options=[False, True],
        )
        parameter_n_jobs = st.sidebar.selectbox(
            "Number of jobs to run in parallel (n_jobs)", options=[1, -1]
        )


st.subheader("1.Dataset")
