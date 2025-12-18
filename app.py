import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="NexGen Predictive Delivery Command Center",
    layout="wide"
)

st.title("NexGen Logistics - Predictive Delivery Command Center")
st.subheader("From reactive firefighting to predictive intelligence")


# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
data = load_data()

if not data or "orders" not in data:
    st.stop()


# -------------------------------------------------
# ORDERS DATA PREP
# -------------------------------------------------
orders_df = data["orders"].copy()
orders_df.columns = orders_df.columns.str.lower().str.strip()

# -------------------------------------------------
# SIDEBAR FILTERS
# -------------------------------------------------
st.sidebar.header("Filters")

priority_filter = st.sidebar.multiselect(
    "Delivery Priority",
    options=orders_df["priority"].dropna().unique()
)

origin_filter = st.sidebar.multiselect(
    "Origin Warehouse",
    options=orders_df["origin"].dropna().unique()
)

if priority_filter:
    orders_df = orders_df[orders_df["priority"].isin(priority_filter)]

if origin_filter:
    orders_df = orders_df[orders_df["origin"].isin(origin_filter)]

st.markdown("### Orders Snapshot")
st.dataframe(orders_df, use_container_width=True)


# -------------------------------------------------
# MERGE DELIVERY PERFORMANCE
# -------------------------------------------------
delivery_df = data["delivery"].copy()
delivery_df.columns = delivery_df.columns.str.lower().str.strip()

merged_df = pd.merge(
    orders_df,
    delivery_df,
    on="order_id",
    how="left"
)


# -------------------------------------------------
# MERGE ROUTES DATA
# -------------------------------------------------
routes_df = data["routes"].copy()
routes_df.columns = routes_df.columns.str.lower().str.strip()

merged_df = pd.merge(
    merged_df,
    routes_df,
    on="order_id",
    how="left"
)


# -------------------------------------------------
# BASIC DATA SANITY (IMPORTANT)
# -------------------------------------------------
merged_df["priority"] = merged_df["priority"].fillna("Unknown")
merged_df["carrier"] = merged_df["carrier"].fillna("Unknown")


# -------------------------------------------------
# DELAY METRICS
# -------------------------------------------------
merged_df["delay_days"] = (
    merged_df["actual_delivery_days"] - merged_df["promised_delivery_days"]
)

merged_df["delay_days"] = merged_df["delay_days"].fillna(0)
merged_df["delayed"] = (merged_df["delay_days"] > 0).astype(int)
merged_df["delay_minutes"] = merged_df["delay_days"] * 24 * 60


# -------------------------------------------------
# EXECUTIVE KPIs
# -------------------------------------------------
st.markdown("## Executive Overview")

total_orders = merged_df["order_id"].nunique()
delayed_orders = merged_df["delayed"].sum()
delay_rate = (delayed_orders / total_orders) * 100 if total_orders else 0
avg_rating = merged_df["customer_rating"].mean()

c1, c2, c3, c4 = st.columns(4)

c1.metric("Total Orders", total_orders)
c2.metric("Delayed Orders", delayed_orders)
c3.metric("Delay Rate (%)", f"{delay_rate:.1f}")
c4.metric("Avg Customer Rating", f"{avg_rating:.2f}")


# -------------------------------------------------
# DELAY INSIGHTS
# -------------------------------------------------
st.markdown("## Delay Insights")

delay_by_priority = (
    merged_df.groupby("priority")["delayed"]
    .mean()
    .reset_index()
)

fig1 = px.bar(
    delay_by_priority,
    x="priority",
    y="delayed",
    title="Delay Rate by Delivery Priority",
    labels={"delayed": "Delay Rate"}
)
st.plotly_chart(fig1, use_container_width=True)


carrier_delay = (
    merged_df.groupby("carrier")["delayed"]
    .mean()
    .reset_index()
    .sort_values("delayed", ascending=False)
)

fig2 = px.bar(
    carrier_delay,
    x="carrier",
    y="delayed",
    title="Delay Rate by Carrier",
    labels={"delayed": "Delay Rate"}
)
st.plotly_chart(fig2, use_container_width=True)


fig3 = px.scatter(
    merged_df,
    x="delay_days",
    y="customer_rating",
    color="priority",
    title="Impact of Delivery Delay on Customer Ratings"
)
st.plotly_chart(fig3, use_container_width=True)


# -------------------------------------------------
# RULE-BASED DELAY RISK SCORE
# -------------------------------------------------
merged_df["priority_risk"] = merged_df["priority"].map({
    "Express": 1.0,
    "Standard": 0.6,
    "Economy": 0.3
}).fillna(0.5)

merged_df["carrier_risk"] = merged_df["carrier"].map(
    carrier_delay.set_index("carrier")["delayed"]
).fillna(0)

merged_df["duration_risk"] = (
    merged_df["promised_delivery_days"] /
    merged_df["promised_delivery_days"].max()
)

merged_df["delay_risk_score"] = (
    0.4 * merged_df["priority_risk"] +
    0.4 * merged_df["carrier_risk"] +
    0.2 * merged_df["duration_risk"]
) * 100


# -------------------------------------------------
# HIGH-RISK ORDERS (RULE-BASED)
# -------------------------------------------------
st.markdown("## High-Risk Orders (Rule-Based)")

risk_threshold = st.slider(
    "Risk Threshold",
    min_value=0,
    max_value=100,
    value=60
)

high_risk_orders = merged_df[
    merged_df["delay_risk_score"] >= risk_threshold
][[
    "order_id",
    "priority",
    "carrier",
    "origin",
    "destination",
    "delay_risk_score"
]].sort_values("delay_risk_score", ascending=False)

st.dataframe(high_risk_orders, use_container_width=True)


# -------------------------------------------------
# MACHINE LEARNING MODEL
# -------------------------------------------------
ml_df = merged_df.loc[:, [
    "priority",
    "carrier",
    "promised_delivery_days",
    "delay_risk_score",
    "delayed"
]].dropna().copy()

# Fill again for absolute safety
ml_df["priority"] = ml_df["priority"].fillna("Unknown")
ml_df["carrier"] = ml_df["carrier"].fillna("Unknown")

priority_encoder = LabelEncoder()
carrier_encoder = LabelEncoder()

# Fit encoders on FULL merged dataset (prevents unseen-label crash)
priority_encoder.fit(merged_df["priority"])
carrier_encoder.fit(merged_df["carrier"])

ml_df["priority_enc"] = priority_encoder.transform(ml_df["priority"])
ml_df["carrier_enc"] = carrier_encoder.transform(ml_df["carrier"])

X = ml_df[[
    "priority_enc",
    "carrier_enc",
    "promised_delivery_days",
    "delay_risk_score"
]]

y = ml_df["delayed"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


@st.cache_resource
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


model = train_model(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.markdown("##  ML Delay Prediction Model")
st.metric("Model Accuracy", f"{accuracy * 100:.1f}%")


# -------------------------------------------------
# ML PREDICTIONS (SAFE)
# -------------------------------------------------
merged_df["priority_enc"] = priority_encoder.transform(
    merged_df["priority"]
)

merged_df["carrier_enc"] = carrier_encoder.transform(
    merged_df["carrier"]
)

merged_df["predicted_delay_prob"] = model.predict_proba(
    merged_df[[
        "priority_enc",
        "carrier_enc",
        "promised_delivery_days",
        "delay_risk_score"
    ]].fillna(0)
)[:, 1]


# -------------------------------------------------
# ML HIGH-RISK ORDERS
# -------------------------------------------------
st.markdown("## ML-Identified At-Risk Orders")

ml_threshold = st.slider(
    "ML Delay Probability Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.6
)

ml_risk_orders = merged_df[
    merged_df["predicted_delay_prob"] >= ml_threshold
][[
    "order_id",
    "priority",
    "carrier",
    "origin",
    "destination",
    "predicted_delay_prob"
]].sort_values("predicted_delay_prob", ascending=False)

st.dataframe(ml_risk_orders, use_container_width=True)

# -------------------------------------------------
# COST IMPACT CALCULATOR
# -------------------------------------------------
st.markdown("## 💰 Cost Impact Analysis")

# Parameters (adjustable assumptions)
avoidable_cost_pct = st.slider(
    "Avoidable Cost per Delayed Order (%)",
    min_value=5,
    max_value=50,
    value=20
)

intervention_success_rate = st.slider(
    "Intervention Success Rate (%)",
    min_value=30,
    max_value=100,
    value=70
)

# Work on ML-identified high-risk orders
cost_df = merged_df[
    merged_df["predicted_delay_prob"] >= ml_threshold
].copy()

# Handle missing delivery cost
cost_df["delivery_cost_inr"] = cost_df["delivery_cost_inr"].fillna(
    cost_df["delivery_cost_inr"].median()
)

# Estimated avoidable cost per order
cost_df["avoidable_cost"] = (
    cost_df["delivery_cost_inr"] *
    (avoidable_cost_pct / 100) *
    (intervention_success_rate / 100)
)

total_avoidable_cost = cost_df["avoidable_cost"].sum()

col1, col2, col3 = st.columns(3)

col1.metric("High-Risk Orders", len(cost_df))
col2.metric("Avg Avoidable Cost / Order (₹)", f"{cost_df['avoidable_cost'].mean():,.0f}")
col3.metric("Total Potential Savings (₹)", f"{total_avoidable_cost:,.0f}")

fig_cost = px.bar(
    cost_df.sort_values("avoidable_cost", ascending=False).head(10),
    x="order_id",
    y="avoidable_cost",
    title="Top 10 Orders by Potential Cost Savings",
    labels={"avoidable_cost": "Potential Savings (₹)"}
)

st.plotly_chart(fig_cost, use_container_width=True)
