# NexGen Predictive Delivery Command Center

A data-driven logistics intelligence platform that enables **predictive, proactive decision-making** for delivery operations.

Built as part of the **Logistics Innovation Challenge**, this project transforms NexGen Logistics from reactive issue handling to a **predictive operations model** using analytics, explainable risk scoring, and machine learning.

---

## Problem Statement

NexGen Logistics operates across multiple warehouses, carriers, and delivery priorities. Despite steady growth, the company faces:

- Delivery delays detected only after failures occur  
- Rising operational and penalty costs  
- Inconsistent customer experience  
- Limited use of existing data for decision-making  

Traditional dashboards answer *“what went wrong?”* but fail to answer:

> **“Which orders are likely to fail next — and what should we do now?”**

---

##  Solution Overview

The **Predictive Delivery Command Center** is an interactive Streamlit application that:

- Predicts delivery delays **before they happen**
- Identifies **high-risk orders** and **unreliable carriers**
- Quantifies **₹ cost leakage**
- Enables **data-backed operational interventions**

The platform combines **rule-based intelligence** (for transparency) with **machine learning** (for predictive accuracy).

---

##  Key Features

###  Executive Dashboard
- Total orders, delayed orders, delay rate  
- Average customer rating  
- Real-time operational visibility  

###  Explainable Delay Risk Scoring
- Priority impact  
- Carrier reliability  
- Delivery duration  
- Produces a **0–100 delay risk score**

###  Machine Learning Prediction
- Random Forest model predicts delay probability  
- ML complements rule-based scoring  
- Cached for performance and stability  

###  At-Risk Order Identification
- Rule-based high-risk orders  
- ML-based high-risk orders  
- Adjustable thresholds for operations teams  

###  Cost Impact Calculator
- Estimates avoidable cost from proactive intervention  
- Adjustable assumptions (success rate, avoidable cost %)  
- Converts predictions into **₹ business value**

###  Interactive Visualizations
- Delay rate by delivery priority  
- Carrier performance comparison  
- Delay vs customer rating impact  
- Cost leakage concentration  

---

##  Datasets Used

The application integrates multiple real-world logistics datasets:

- **orders.csv** – Order details, priority, value, origin, destination  
- **delivery_performance.csv** – Promised vs actual delivery, ratings, costs  
- **routes_distance.csv** – Distance, traffic delays, route factors  
- **vehicle_fleet.csv** – Fleet characteristics  
- **warehouse_inventory.csv** – Inventory distribution  
- **customer_feedback.csv** – Ratings and feedback  
- **cost_breakdown.csv** – Detailed cost components  

> Not all orders have complete data, reflecting real operational conditions.

---

##  System Architecture
Raw Data
↓
Data Cleaning & Normalization
↓
Feature Engineering (Delay, Risk, Cost)
↓
Rule-Based Risk Scoring
↓
Machine Learning Prediction
↓
Operational Dashboards & Insights


---

## Business Impact

- **15–20% potential reduction** in delay-related operational costs  
- Improved customer experience through proactive handling  
- Better carrier selection for high-priority orders  
- Shift from reactive firefighting to predictive operations  

The platform enables NexGen to prioritize actions based on **risk + ROI**, not intuition.

---

## Tech Stack

- **Python**
- **Streamlit** – Interactive dashboard  
- **Pandas / NumPy** – Data processing  
- **Plotly** – Interactive visualizations  
- **Scikit-learn** – Machine learning (Random Forest)

---

##  How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone <your-repo-url>
cd nexgen-logistics
```
### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run the application
```bash
streamlit run app.py
The application will open automatically in your default web browser.
```
---

##  Future Enhancements

- Automated carrier reassignment  
- What-if simulation (carrier / priority swap)  
- Carbon footprint & sustainability scoring  
- Real-time alerts and notifications  
- Integration with warehouse optimization systems  

---

##  Conclusion

The **NexGen Predictive Delivery Command Center** demonstrates how logistics data can be transformed into **actionable intelligence**.

By combining analytics, explainable scoring, and machine learning, the solution positions NexGen Logistics as an **innovation-led, future-ready logistics provider**.


