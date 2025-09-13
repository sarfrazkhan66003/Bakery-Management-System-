# app.py
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
from datetime import datetime, date
import pickle
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import io
import plotly.express as px
import os

DB_FILE = "bakery_full.db"
MODEL_FILE = "customer_seg_model.pkl"

# --------------------------
# Database Manager (OOP)
# --------------------------
class DatabaseManager:
    def __init__(self, db_file=DB_FILE):
        self.db_file = db_file
        # allow multi-thread in streamlit
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self._initialize_tables()

    def _initialize_tables(self):
        cur = self.conn.cursor()
        # customers
        cur.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                phone TEXT,
                gender TEXT,
                age INTEGER,
                city TEXT,
                loyalty INTEGER DEFAULT 0,
                signup_date TEXT
            )
        """)
        # products (inventory)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE,
                category TEXT,
                stock INTEGER,
                price REAL
            )
        """)
        # bills (summary)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS bills (
                bill_id INTEGER PRIMARY KEY AUTOINCREMENT,
                customer_id INTEGER,
                customer_name TEXT,
                customer_phone TEXT,
                total_amount REAL,
                discount REAL,
                gst REAL,
                net_amount REAL,
                delivery_method TEXT, -- 'Delivery' or 'Pickup'
                address TEXT,
                bill_date TEXT
            )
        """)
        # order_lines (details per bill)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS order_lines (
                order_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bill_id INTEGER,
                product_id INTEGER,
                product_name TEXT,
                quantity INTEGER,
                unit_price REAL,
                line_total REAL
            )
        """)
        # transactions table (for RFM & history) - duplicate from order_lines + bill ref
        cur.execute("""
            CREATE TABLE IF NOT EXISTS transactions (
                tx_id INTEGER PRIMARY KEY AUTOINCREMENT,
                bill_id INTEGER,
                customer_id INTEGER,
                tx_date TEXT,
                product_id INTEGER,
                product_name TEXT,
                quantity INTEGER,
                net_amount REAL
            )
        """)
        self.conn.commit()

    def execute(self, query, params=(), fetch=False):
        cur = self.conn.cursor()
        cur.execute(query, params)
        if fetch:
            rows = cur.fetchall()
            return rows
        else:
            self.conn.commit()
            return None

    def to_dataframe(self, query):
        return pd.read_sql_query(query, self.conn)

# --------------------------
# Product Manager
# --------------------------
class ProductManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add_or_update_product(self, name, category, stock, price):
        # try insert, on conflict update
        try:
            self.db.execute("""
                INSERT INTO products (name, category, stock, price)
                VALUES (?, ?, ?, ?)
            """, (name, category, int(stock), float(price)))
        except Exception:
            # update existing
            self.db.execute("""
                UPDATE products SET category=?, stock=?, price=?
                WHERE name=?
            """, (category, int(stock), float(price), name))
        return True

    def get_all_products_df(self):
        return self.db.to_dataframe("SELECT * FROM products")

    def get_product_by_id(self, pid):
        rows = self.db.execute("SELECT product_id, name, stock, price FROM products WHERE product_id=?", (pid,), fetch=True)
        return rows[0] if rows else None

    def get_product_by_name(self, name):
        rows = self.db.execute("SELECT product_id, name, stock, price FROM products WHERE name=?", (name,), fetch=True)
        return rows[0] if rows else None

    def reduce_stock(self, product_id, qty):
        self.db.execute("UPDATE products SET stock = stock - ? WHERE product_id = ? AND stock >= ?",
                        (int(qty), int(product_id), int(qty)))

# --------------------------
# Customer Manager
# --------------------------
class CustomerManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def add_customer(self, name, phone, gender=None, age=None, city=None, loyalty=0):
        signup = date.today().isoformat()
        self.db.execute("""
            INSERT INTO customers (name, phone, gender, age, city, loyalty, signup_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (name, phone, gender, age, city, int(loyalty), signup))

    def update_customer(self, customer_id, **kwargs):
        # simple updater for provided fields
        fields = []
        vals = []
        for k, v in kwargs.items():
            fields.append(f"{k}=?")
            vals.append(v)
        vals.append(customer_id)
        q = f"UPDATE customers SET {','.join(fields)} WHERE customer_id=?"
        self.db.execute(q, tuple(vals))

    def get_customers_df(self):
        return self.db.to_dataframe("SELECT * FROM customers")

    def get_customer(self, customer_id):
        rows = self.db.execute("SELECT * FROM customers WHERE customer_id=?", (customer_id,), fetch=True)
        return rows[0] if rows else None

# --------------------------
# Order / Billing Manager
# --------------------------
class BillingManager:
    def __init__(self, db: DatabaseManager, prod_mgr: ProductManager):
        self.db = db
        self.prod = prod_mgr

    def create_bill(self, customer_id, customer_name, customer_phone, items: list,
                    discount_pct=0.0, delivery_method="Pickup", address=""):
        """
        items: list of dicts: [{"product_id": id, "product_name": name, "quantity": q, "unit_price": p}, ...]
        returns: (success, message, bill_id)
        """
        # Validate stock
        for it in items:
            pid = it["product_id"]
            qty = int(it["quantity"])
            prod = self.prod.get_product_by_id(pid)
            if not prod:
                return False, f"Product id {pid} not found", None
            stock = prod[2]  # (product_id, name, stock, price)
            if qty > stock:
                return False, f"Not enough stock for {prod[1]}. Available: {stock}", None

        # compute totals
        subtotal = sum([it["quantity"] * it["unit_price"] for it in items])
        discount = subtotal * (discount_pct / 100.0)
        gst = 0.18 * (subtotal - discount)  # 18% GST
        net = subtotal - discount + gst

        bill_date = datetime.now().isoformat()
        # insert bill
        self.db.execute("""
            INSERT INTO bills (customer_id, customer_name, customer_phone, total_amount, discount, gst, net_amount, delivery_method, address, bill_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (customer_id, customer_name, customer_phone, subtotal, discount, gst, net, delivery_method, address, bill_date))

        bill_id = self.db.execute("SELECT last_insert_rowid()", fetch=True)[0][0]

        # insert lines and reduce stock, create transactions
        for it in items:
            pid = it["product_id"]
            name = it["product_name"]
            qty = int(it["quantity"])
            unit = float(it["unit_price"])
            line_total = qty * unit
            self.db.execute("""
                INSERT INTO order_lines (bill_id, product_id, product_name, quantity, unit_price, line_total)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (bill_id, pid, name, qty, unit, line_total))
            # reduce stock
            self.prod.reduce_stock(pid, qty)
            # insert transaction
            self.db.execute("""
                INSERT INTO transactions (bill_id, customer_id, tx_date, product_id, product_name, quantity, net_amount)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (bill_id, customer_id, bill_date, pid, name, qty, line_total))

        return True, "Bill created", bill_id

    def get_bill_df(self, bill_id):
        return self.db.to_dataframe(f"SELECT * FROM order_lines WHERE bill_id = {int(bill_id)}")

    def get_bills_df(self):
        return self.db.to_dataframe("SELECT * FROM bills ORDER BY bill_date DESC")

    def get_transactions_df(self):
        return self.db.to_dataframe("SELECT * FROM transactions ORDER BY tx_date DESC")

# --------------------------
# Analytics Manager
# --------------------------
class AnalyticsManager:
    def __init__(self, db: DatabaseManager):
        self.db = db

    def monthly_sales(self):
        df = self.db.to_dataframe("SELECT bill_date, net_amount FROM bills")
        if df.empty:
            return pd.DataFrame(columns=["month","sales"])
        df["bill_date"] = pd.to_datetime(df["bill_date"])
        df["month"] = df["bill_date"].dt.to_period("M").astype(str)
        m = df.groupby("month")["net_amount"].sum().reset_index().rename(columns={"net_amount":"sales"})
        return m

    def top_customers(self, n=10):
        df = self.db.to_dataframe("""
            SELECT customer_name, SUM(net_amount) as total_spent, COUNT(bill_id) as bills
            FROM bills
            GROUP BY customer_name
            ORDER BY total_spent DESC
            LIMIT %d
        """ % n)
        return df

    def stock_levels(self):
        return self.db.to_dataframe("SELECT name, stock FROM products ORDER BY stock ASC")

# --------------------------
# ML Manager (RFM + KMeans)
# --------------------------
class MLManager:
    def __init__(self, db: DatabaseManager, model_file=MODEL_FILE):
        self.db = db
        self.model_file = model_file

    def build_rfm(self):
        # build RFM table from transactions / bills
        cust_df = self.db.to_dataframe("SELECT customer_id, name FROM customers")
        bills = self.db.to_dataframe("SELECT bill_id, customer_id, net_amount, bill_date FROM bills")
        if bills.empty:
            return pd.DataFrame()
        bills["bill_date"] = pd.to_datetime(bills["bill_date"])
        today = pd.to_datetime(datetime.now())
        rfm = bills.groupby("customer_id").agg(
            frequency=("bill_id", "count"),
            monetary=("net_amount", "sum"),
            last_purchase=("bill_date", "max")
        ).reset_index()
        rfm["recency"] = (today - rfm["last_purchase"]).dt.days
        rfm = rfm.merge(cust_df, on="customer_id", how="left")
        # fill
        rfm = rfm.fillna({"frequency":0, "monetary":0, "recency":999})
        return rfm[["customer_id","name","frequency","recency","monetary"]]

    def train_kmeans(self, n_clusters=3):
        rfm = self.build_rfm()
        if rfm.empty:
            return False, "Not enough data to train", None
        X = rfm[["frequency","recency","monetary"]].values
        # scale
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        k = min(n_clusters, max(1, len(rfm)))
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = model.fit_predict(Xs)
        rfm["segment"] = labels
        # save model + scaler together
        with open(self.model_file, "wb") as f:
            pickle.dump({"model":model, "scaler":scaler}, f)
        return True, "Model trained & saved", rfm

    def predict_segment(self, frequency, recency, monetary):
        if not os.path.exists(self.model_file):
            return None, "Model not found"
        with open(self.model_file, "rb") as f:
            data = pickle.load(f)
        model = data["model"]
        scaler = data["scaler"]
        X = np.array([[frequency, recency, monetary]])
        Xs = scaler.transform(X)
        label = int(model.predict(Xs)[0])
        return label, "OK"

# --------------------------
# Utility: DataFrame -> Excel bytes
# --------------------------
def df_dict_to_excel_bytes(dfs: dict):
    """
    dfs: {"sheet1": df1, ...}
    returns bytes for download
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet, df in dfs.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
    return output.getvalue()

# --------------------------
# STREAMLIT UI
# --------------------------
st.set_page_config(page_title="Bakery Management + Billing + Segmentation", layout="wide")
st.title("🥐 Bakery Management System — Inventory / Billing / Segmentation")

# initialize managers
dbm = DatabaseManager()
prod_mgr = ProductManager(dbm)
cust_mgr = CustomerManager(dbm)
bill_mgr = BillingManager(dbm, prod_mgr)
analytics = AnalyticsManager(dbm)
ml = MLManager(dbm)

# Sidebar: quick utilities
st.sidebar.header("Utilities")
if st.sidebar.button("Export DB to Excel"):
    try:
        prods = prod_mgr.get_all_products_df()
    except Exception:
        prods = pd.DataFrame()
    try:
        custs = cust_mgr.get_customers_df()
    except Exception:
        custs = pd.DataFrame()
    try:
        bills = bill_mgr.get_bills_df()
        orders = bill_mgr.get_transactions_df()
    except Exception:
        bills = pd.DataFrame()
        orders = pd.DataFrame()

    xls = df_dict_to_excel_bytes({"products":prods, "customers":custs, "bills":bills, "transactions":orders})
    st.sidebar.download_button("Download DB Excel", xls, "bakery_db_export.xlsx")

st.sidebar.markdown("---")
role = st.sidebar.selectbox("Role", ["Customer / Counter", "Baker / Admin"])
st.sidebar.markdown("Model: " + ("Found" if os.path.exists(MODEL_FILE) else "Not trained"))

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Customers", "Products / Inventory", "Billing Counter", "Customer Segmentation"])

# ---------------- Dashboard ----------------
with tab1:
    st.subheader("Dashboard")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Monthly Sales")
        ms = analytics.monthly_sales()
        if ms.empty:
            st.info("No sales yet")
        else:
            fig = px.bar(ms, x="month", y="sales", title="Monthly Sales")
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### Top Customers")
        topcust = analytics.top_customers(10)
        if topcust.empty:
            st.info("No customers yet")
        else:
            fig2 = px.bar(topcust, x="customer_name", y="total_spent", title="Top Customers")
            st.plotly_chart(fig2, use_container_width=True)

    with col2:
        st.markdown("### Stock Levels (lowest first)")
        stock = analytics.stock_levels()
        if stock.empty:
            st.info("No products")
        else:
            fig3 = px.bar(stock, x="name", y="stock", title="Stock Levels")
            st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")
    st.markdown("Recent Bills")
    try:
        bills_df = bill_mgr.get_bills_df()
        st.dataframe(bills_df.head(20))
    except Exception:
        st.info("No bills yet")

# ---------------- Customers ----------------
with tab2:
    st.subheader("Customer Management")
    st.write("Add new customer")
    with st.form("add_customer_form"):
        cname = st.text_input("Name")
        phone = st.text_input("Phone")
        gender = st.selectbox("Gender", ["", "Male", "Female", "Other"])
        age = st.number_input("Age", min_value=0, max_value=120, value=25)
        city = st.text_input("City")
        loyalty = st.selectbox("Loyalty Member", [0, 1], index=0)
        save_cust = st.form_submit_button("Save Customer")
    if save_cust:
        if not cname or not phone:
            st.error("Name and Phone required")
        else:
            cust_mgr.add_customer(cname.strip(), phone.strip(), gender, int(age), city, int(loyalty))
            st.success("Customer saved")

    st.markdown("### All customers")
    try:
        dfcust = cust_mgr.get_customers_df()
        st.dataframe(dfcust)
    except Exception:
        st.info("No customers")

# ---------------- Products / Inventory ----------------
with tab3:
    st.subheader("Products / Inventory")
    with st.form("add_product_form"):
        pname = st.text_input("Product name")
        pcat = st.text_input("Category")
        pstock = st.number_input("Stock quantity", min_value=0, value=10)
        pprice = st.number_input("Price (per unit)", min_value=0.0, value=50.0)
        save_prod = st.form_submit_button("Add / Update Product")
    if save_prod:
        if not pname:
            st.error("Product name required")
        else:
            prod_mgr.add_or_update_product(pname.strip(), pcat.strip(), int(pstock), float(pprice))
            st.success("Product added / updated")

    st.markdown("### Inventory")
    prod_df = prod_mgr.get_all_products_df()
    if prod_df.empty:
        st.info("No products")
    else:
        st.dataframe(prod_df)

# ---------------- Billing Counter ----------------
with tab4:
    st.subheader("Billing Counter")
    customers_df = cust_mgr.get_customers_df()
    if customers_df.empty:
        st.info("No customers. Please add a customer first.")
    else:
        cust_selection = st.selectbox("Select Customer", customers_df.apply(lambda r: f"{r.customer_id} - {r.name} ({r.phone})", axis=1).tolist())
        selected_id = int(cust_selection.split(" - ")[0])
        selected_customer = dbm.execute("SELECT name, phone FROM customers WHERE customer_id=?", (selected_id,), fetch=True)[0]
        st.markdown(f"**Customer:** {selected_customer[0]}  \n**Phone:** {selected_customer[1]}")

        st.markdown("Choose delivery method")
        delivery = st.selectbox("Delivery or Pickup", ["Pickup", "Delivery"])
        address = ""
        if delivery == "Delivery":
            address = st.text_input("Delivery Address (required for delivery)")

        st.markdown("Add products to bill")
        products_df = prod_mgr.get_all_products_df()
        if products_df.empty:
            st.info("No products to sell. Add products first.")
        else:
            # build order cart
            cart = []
            cols = st.columns([3,1,1,1])
            cols[0].write("Product")
            cols[1].write("Price")
            cols[2].write("In Stock")
            cols[3].write("Qty to add")
            qty_inputs = {}
            for idx, row in products_df.iterrows():
                pid = int(row.product_id)
                name = row.name
                price = float(row.price)
                stock = int(row.stock)
                cols = st.columns([3,1,1,1])
                cols[0].write(name)
                cols[1].write(f"₹{price:.2f}")
                cols[2].write(stock)
                qty = cols[3].number_input(f"qty_{pid}", min_value=0, max_value=stock, value=0, key=f"qty_{pid}")
                if qty > 0:
                    qty_inputs[pid] = {"product_id": pid, "product_name": name, "quantity": int(qty), "unit_price": price}

            discount_pct = st.number_input("Discount % (optional)", min_value=0.0, max_value=100.0, value=0.0)
            if st.button("Generate Bill & Save"):
                if delivery == "Delivery" and not address.strip():
                    st.error("Please enter delivery address.")
                elif not qty_inputs:
                    st.error("Add at least one product with qty > 0")
                else:
                    items = list(qty_inputs.values())
                    success, msg, bill_id = bill_mgr.create_bill(selected_id, selected_customer[0], selected_customer[1],
                                                                 items, discount_pct, delivery, address)
                    if not success:
                        st.error(msg)
                    else:
                        st.success(f"Bill #{bill_id} created successfully")
                        bill_df = bill_mgr.get_bill_df(bill_id)
                        st.markdown("### Bill details")
                        st.table(bill_df.assign(
                            unit_price=lambda d: d["unit_price"].map(lambda x: f"₹{x:.2f}"),
                            line_total=lambda d: d["line_total"].map(lambda x: f"₹{x:.2f}")
                        ))
                        # show bill summary
                        bills = bill_mgr.get_bills_df()
                        this_bill = bills[bills["bill_id"]==bill_id].iloc[0]
                        st.markdown(f"**Subtotal:** ₹{this_bill['total_amount']:.2f}  ")
                        st.markdown(f"**Discount:** ₹{this_bill['discount']:.2f}  ")
                        st.markdown(f"**GST (18%):** ₹{this_bill['gst']:.2f}  ")
                        st.markdown(f"**Net Payable:** ₹{this_bill['net_amount']:.2f}  ")

                        # allow download bill as Excel
                        df_bytes = df_dict_to_excel_bytes({f"Bill_{bill_id}": bill_df})
                        st.download_button("Download Bill (Excel)", df_bytes, f"bill_{bill_id}.xlsx")

# ---------------- Customer Segmentation ----------------
with tab5:
    st.subheader("Customer Segmentation (RFM + KMeans)")
    st.markdown("Build RFM features and train KMeans. Model saved to disk and used for predictions.")
    rfm = ml.build_rfm()
    if rfm.empty:
        st.info("Not enough bill/transaction data to compute RFM. Create some bills first.")
    else:
        st.markdown("RFM Table")
        st.dataframe(rfm)

    st.markdown("Train KMeans model")
    n_clusters = st.number_input("Number of clusters", min_value=2, max_value=10, value=3)
    if st.button("Train / Retrain Model"):
        ok, msg, res = ml.train_kmeans(int(n_clusters))
        if not ok:
            st.error(msg)
        else:
            st.success(msg)
            st.dataframe(res)

    st.markdown("Predict segment for a new/hypothetical customer")
    freq = st.number_input("Frequency (no. of bills)", min_value=0, value=1)
    rec = st.number_input("Recency (days since last purchase)", min_value=0, value=30)
    mon = st.number_input("Monetary (total spent)", min_value=0.0, value=100.0)
    if st.button("Predict Segment"):
        label, m = ml.predict_segment(int(freq), int(rec), float(mon))
        if label is None:
            st.error(m)
        else:
            st.success(f"Predicted segment: {label}")

st.markdown("---")
st.caption("Made with ❤️ — ye starter app hai. Agar chaho to PDF invoice, email notifications, reorder alerts, user auth aur deployment steps bhi add kar doon.")
