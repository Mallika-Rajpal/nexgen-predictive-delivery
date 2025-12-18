import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    data = {}

    try:
        data["orders"] = pd.read_csv("data/orders.csv")
        data["delivery"] = pd.read_csv("data/delivery_performance.csv")
        data["routes"] = pd.read_csv("data/routes_distance.csv")
        data["fleet"] = pd.read_csv("data/vehicle_fleet.csv")
        data["warehouse"] = pd.read_csv("data/warehouse_inventory.csv")
        data["feedback"] = pd.read_csv("data/customer_feedback.csv")
        data["cost"] = pd.read_csv("data/cost_breakdown.csv")
    except Exception as e:
        st.error(f"Error loading data: {e}")

    return data
