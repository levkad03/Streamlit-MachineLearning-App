import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Page Layout

st.set_page_config(page_title="First Streamlit Project", layout="wide")


def build_model(df):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=(100 - split_size) / 100
    )

    st.markdown("**1.2. Data splits**")
    st.write("Training set")
    st.info(X_train.shape)
    st.write("Test set")
    st.info(X_test.shape)

    st.markdown("**1.3. Variable details**:")
    st.write("X variable")
    st.info(list(X.columns))
    st.write("Y variable")
    st.info(y.name)

    rf = RandomForestRegressor(
        n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs,
    )

    rf.fit(X_train, y_train)

    st.subheader("2.Model Performance")
    st.markdown("**2.1. Training set**")
    y_pred_train = rf.predict(X_train)
    st.write("Coefficient of determination ($R^2$):")
    st.info(r2_score(y_train, y_pred_train))

    st.write(f"Error {parameter_metric}:")

    if parameter_metric == "mse":
        st.info(mean_squared_error(y_train, y_pred_train))
    elif parameter_metric == "mae":
        st.info(mean_absolute_error(y_train, y_pred_train))

    st.markdown("**2.2. Test set**")
    y_pred_test = rf.predict(X_test)
    st.write("Coefficient of determination ($R^2$):")
    st.info(r2_score(y_test, y_pred_test))

    st.write(f"Error {parameter_metric}:")
    if parameter_metric == "mse":
        st.info(mean_squared_error(y_test, y_pred_test))
    elif parameter_metric == "mae":
        st.info(mean_absolute_error(y_test, y_pred_test))

    st.subheader("3. Model Parameters")
    st.write(rf.get_params())

    st.subheader("4. Feature Importance")
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": rf.feature_importances_}
    ).sort_values(by="importance", ascending=False)
    fig_importance = px.bar(
        feature_importance,
        x="importance",
        y="feature",
        orientation="h",
        title="Feature Importance",
    )
    st.plotly_chart(fig_importance)

    st.subheader("5. Error Plots")
    error_train = y_train - y_pred_train
    error_test = y_test - y_pred_test

    fig_train_error = px.histogram(
        error_train, nbins=30, title="Training Error Distribution"
    )
    fig_test_error = px.histogram(error_test, nbins=30, title="Test Error Distribution")

    st.plotly_chart(fig_train_error)
    st.plotly_chart(fig_test_error)


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
    parameter_criterion = st.sidebar.selectbox(
        "Quality of a split (Criterion)",
        options=["squared_error", "absolute_error", "friedman_mse", "poisson"],
    )

    with st.sidebar.subheader("2.2. General Parameters"):
        parameter_random_state = st.sidebar.slider(
            "Seed number (random_state)", 0, 1000, 42, 1
        )
        parameter_metric = st.sidebar.selectbox(
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


if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown("**1.1. Glimpse of dataset**")
    st.write(df)
    build_model(df)
else:
    st.info("Awaiting for CSV file to be uploaded.")
    if st.button("Press to use Example Dataset"):
        diabetes = load_diabetes()
        X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        Y = pd.Series(diabetes.target, name="response")
        df = pd.concat([X, Y], axis=1)

        st.markdown("The Diabetes dataset is used as the example.")
        st.write(df.head(5))

        build_model(df)
