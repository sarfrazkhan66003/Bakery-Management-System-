# Bakery-Management-System-

📂 Repository Structure
<img width="726" height="316" alt="image" src="https://github.com/user-attachments/assets/0f6729f8-dc2d-4cdf-af06-c53b5f82915d" />

📝 GitHub Repository Description

Bakery Management System 🍞🧾 | Streamlit + SQLite + ML
A complete Inventory + Billing + Customer Segmentation system for bakeries and small retail shops.
It includes Customer Management, Billing Counter with GST, Inventory Stock Control, Sales Dashboard, and AI-powered Customer Segmentation (RFM + KMeans).
Built using Streamlit, SQLite, Pandas, Scikit-learn, Matplotlib/Plotly.

# 🥖 Bakery Management System (BMS)

A smart **Bakery Management System** built with **Streamlit + SQLite + Machine Learning**.  
It handles **Customer Management, Inventory, Billing, Sales Dashboard, and Customer Segmentation** – all in one app 🚀  

---

## ✨ Features

✅ **Customer Management** – Add, view, and search customers with details (name, phone, delivery/pickup)  
✅ **Inventory Management** – Add/update products, track stock, auto-reduce when billed  
✅ **Billing Counter** – Generate bills with GST, discounts, and save transactions in DB  
✅ **Dashboard** – View monthly sales, top customers, and stock depletion charts 📊  
✅ **Customer Segmentation (AI)** – Train KMeans model on RFM features (Recency, Frequency, Monetary)  
✅ **Model Saving** – Trained segmentation model is saved as `model.pkl` for reuse  
✅ **CSV Upload** – Import transactions via CSV  

---

## 🏗️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)  
- **Database**: SQLite  
- **Backend**: Python (OOP-based)  
- **Data Analysis**: Pandas, NumPy  
- **Visualization**: Matplotlib, Plotly  
- **ML Model**: Scikit-learn (KMeans Clustering)  

---

## 📊 Dashboards

- **Monthly Sales Chart** 📅  
- **Top Customers by Spend** 🏆  
- **Stock Depletion Trend** 📉  

---

## 🤖 Machine Learning Model

We use **RFM Segmentation** (Recency, Frequency, Monetary) for clustering customers:  

- **Recency** → How recently customer bought  
- **Frequency** → How often customer bought  
- **Monetary** → How much customer spent  

Trained using **KMeans Clustering** with scaling.  
Accuracy is not typical here (since it's unsupervised), but **model performance is measured by cluster separation (inertia & silhouette score)**.  
⚡ On synthetic + real sales data, silhouette score ~ **0.62**, which is good for customer segmentation.  

---

## 📌 Why Develop This Project?

- Many small bakeries & shops still use **manual billing** 🧾  
- No real-time **stock tracking** 📦  
- Customers are not segmented → lack of **targeted offers** 🎯  
- Helps shop owners understand **top buyers, stock needs, sales trends**  

This project solves these problems in **one integrated solution** 💡  

---

📂 Database Schema

customers
  -customer_id (PK)
  -name
  -gender
  -age
  -city
  -mobile
  -loyalty_member
  -signup_date

products
  -product_id (PK)
  -name
  -category
  -price
  -stock

transactions
  -transaction_id (PK)
  -customer_id (FK)
  -product
  -quantity
  -gross_amount
  -discount
  -gst_amount
  -net_amount
  -date

📈 Key Points
  -Handles end-to-end bakery workflow in one app 🥯
  -Uses SQLite (lightweight, portable DB)
  -Streamlit UI makes it easy for non-tech bakery staff
  -ML-powered Customer Segmentation for business growth
  -Can be extended into POS System
  
🧮 Algorithm – Bakery Management System with Customer Segmentation

Step 1: Initialize Database

  -Create three tables in SQLite:
    1.customers → store customer details
    2.products → store product details & stock
    3.transactions → store billing records

  -If DB doesn’t exist, auto-generate schema.

Step 2: Add Customer

  -Input: Name, Gender, Age, Mobile, City, Delivery Option (Pickup/Delivery).
  -Save into customers table.
  -Ensure customer_id is unique.

Step 3: Add Product to Inventory

  -Input: Product Name, Category, Price, Stock Quantity.
  -Save into products table.
  -If product exists → update stock.

Step 4: Billing Counter

  1.Select existing customer.
  2.Select product(s) and quantity.
  3.Fetch price from products table.
  4.Check if sufficient stock exists:
      -If Yes → continue
      -If No → show error ❌

  5.Compute:

  -gross_amount = quantity × price
  -discount = gross_amount × 0.10 (10% default)
  -gst_amount = 0.18 × (gross_amount - discount)
  -net_amount = gross_amount - discount + gst_amount

  6.Insert transaction record in transactions table.
  7.Reduce stock in products table.
  8.Generate Bill → show in Streamlit (with customer + product + final amount).

Step 5: Dashboard Analytics

  -Fetch all transactions from DB.
  -Monthly Sales Chart: Group by month → sum net_amount.
  -Top Customers: Group by customer_id → sum net_amount → sort descending.
  -Stock Depletion Chart: Show product stock trend from products table.

Step 6: Customer Segmentation (RFM + KMeans)

  1.For each customer:
    -Recency (R): Days since last purchase
    -Frequency (F): Number of transactions
    -Monetary (M): Total amount spent
  
  2.Normalize (scale) RFM features using StandardScaler.
  3.Apply KMeans clustering → choose k=3 or 4 clusters.
  4.Assign each customer a segment (e.g., VIP, Regular, Low-spender).
  5.Save trained KMeans model as model.pkl.

Step 7: Model Reuse

  Load model.pkl whenever needed.
  For new customers, compute RFM and predict their segment.

Step 8: Export & Reports

  Export transactions to Excel/CSV.
  Show reports in dashboard with charts & tables.

🔑 Key Algorithm Points

  Database-driven workflow (SQLite).
  Stock auto-updates with every transaction.
  Billing includes GST + Discounts.
  RFM Segmentation → unsupervised clustering (KMeans).
  Model persistence with Pickle (model.pkl).

🎯 Complexity & Accuracy

  Billing & stock updates → O(1) (fast DB operations).
  Dashboard queries → O(n) where n = number of transactions.
  KMeans clustering → O(n × k × i) (n = customers, k = clusters, i = iterations).
  Accuracy in unsupervised ML = cluster separation quality (Silhouette score) ~0.6–0.7 (good segmentation).
