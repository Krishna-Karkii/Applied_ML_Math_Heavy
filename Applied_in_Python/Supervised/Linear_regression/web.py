import streamlit as st
import matplotlib.pyplot as pt
import numpy as np
import pandas as pd

from linear_regression import LinearRegressionStochastic, run_regression, min_max
from pca import reduce_dim_pca

st.title("Applied Linear Regression on House Prices.")
st.space("xxsmall")

epoch = st.text_input("Enter epoch for the Linear regression.")

data = pd.read_csv("Applied_in_Python/Supervised/Linear_regression/data.csv")

lr = None
processed_d = None

if lr not in st.session_state:
    st.session_state.lr = None
if processed_d not in st.session_state:
    st.session_state.processed_d = None


data, df_min, df_max, df_mean, lr = run_regression(data, 4, 4)
lr._gradient_descent()

st.session_state.lr = lr
st.session_state.processed_d = data
print("Executed")

st.space("xxsmall")

if st.session_state.lr is not None:
    projected_d, projected_l = reduce_dim_pca(st.session_state.processed_d, st.session_state.lr.params)
    fig, ax = pt.subplots()
    ax.scatter(projected_d[0], projected_d[1], alpha=0.5, label="Data points")
    print(projected_l[0], projected_l[1])
    ax.plot(projected_l, color='red', label="High-Dim Line", lw=2)
    ax.legend()
    st.pyplot(fig)