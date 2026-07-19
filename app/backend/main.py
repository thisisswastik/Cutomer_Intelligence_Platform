import sqlite3
import pickle
import warnings
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
import pandas as pd
import numpy as np
import joblib
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response

warnings.filterwarnings("ignore")

# Define paths
backend_dir = Path(__file__).resolve().parent
project_root = backend_dir.parents[1]
db_path = project_root / "datasets" / "precomputed" / "olist_dashboard.db"
svd_model_path = project_root / "datasets" / "precomputed" / "svd_model.pkl"
frontend_dir = project_root / "app" / "frontend"

app = FastAPI(title="Customer Intelligence Platform API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables to cache model and items
svd_model = None
candidate_products = []

teleco_model_path = project_root / "notebooks" / "teleco" / "final_churn_xgb_model.pkl"
teleco_model = None
teleco_scaler = None
teleco_feature_names = None

# Instacart global cache (lazy-loaded on first request)
instacart_cache = None
instacart_data_dir = project_root / "datasets" / "instacart_market_ basket"

@app.on_event("startup")
def load_assets():
    global svd_model, candidate_products, teleco_model, teleco_scaler, teleco_feature_names
    # Load SVD model
    if svd_model_path.exists():
        try:
            with open(svd_model_path, "rb") as f:
                svd_model = pickle.load(f)
            print("Successfully loaded SVD recommendation model.")
        except Exception as e:
            print(f"Error loading SVD model: {e}")
    else:
        print("SVD model pickle not found. Dynamic SVD recommendations will be unavailable.")

    # Load candidate products list from DB
    if db_path.exists():
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT product_id FROM product_similarity_top")
            candidate_products = [row[0] for row in cursor.fetchall()]
            # If empty, try to get from purchase history
            if not candidate_products:
                cursor.execute("SELECT DISTINCT product_id FROM customer_purchase_history")
                candidate_products = [row[0] for row in cursor.fetchall()]
            conn.close()
            print(f"Loaded {len(candidate_products)} candidate products for recommendation.")
        except Exception as e:
            print(f"Error loading candidate products: {e}")

    # Load Teleco churn model
    if teleco_model_path.exists():
        try:
            teleco_payload = joblib.load(teleco_model_path)
            teleco_model = teleco_payload['model']
            teleco_scaler = teleco_payload['scaler']
            teleco_feature_names = teleco_payload['feature_names']
            print("Successfully loaded Teleco churn model.")
        except Exception as e:
            print(f"Error loading Teleco churn model: {e}")
    else:
        print("Teleco churn model not found at startup. Will try lazy loading when requested.")

@app.post("/api/precompute/trigger")
def trigger_precomputation():
    try:
        precompute_script = backend_dir / "precompute.py"
        print(f"Triggering precomputation script: {precompute_script}")
        result = subprocess.run(
            [sys.executable, str(precompute_script)],
            capture_output=True,
            text=True,
            check=True
        )
        load_assets()
        return {"status": "success", "output": result.stdout}
    except subprocess.CalledProcessError as e:
        print(f"Precomputation subprocess error: {e.stderr}")
        raise HTTPException(status_code=500, detail=f"Precomputation failed: {e.stderr}")
    except Exception as e:
        print(f"Unexpected error triggering precomputation: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

# Helper function to get DB connection
def get_db_conn():
    if not db_path.exists():
        raise HTTPException(status_code=500, detail="Dashboard database file not found. Please run precomputation.")
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    return conn

# Instacart lazy-loader: reads + caches downsampled dataset (user_id <= 5000) on first call
def get_instacart_data():
    global instacart_cache
    if instacart_cache is not None:
        return instacart_cache
    if not instacart_data_dir.exists():
        return None
    print("[Instacart] Loading dataset (user_id <= 5000) — this runs once...")
    orders = pd.read_csv(instacart_data_dir / "orders.csv")
    orders = orders[orders["user_id"] <= 5000]
    order_ids = set(orders["order_id"])

    op_prior = pd.read_csv(instacart_data_dir / "order_products__prior.csv")
    op_prior = op_prior[op_prior["order_id"].isin(order_ids)]

    products = pd.read_csv(instacart_data_dir / "products.csv")
    aisles   = pd.read_csv(instacart_data_dir / "aisles.csv")
    departments = pd.read_csv(instacart_data_dir / "departments.csv")

    product_details = (
        products
        .merge(aisles, on="aisle_id")
        .merge(departments, on="department_id")
    )
    df = op_prior.merge(product_details, on="product_id").merge(orders, on="order_id")

    instacart_cache = {"orders": orders, "df": df}
    print("[Instacart] Dataset cached successfully.")
    return instacart_cache

# --- API ENDPOINTS ---

@app.get("/api/kpi/overview")
def get_overview_kpis():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # Overview metrics
    cursor.execute("SELECT * FROM executive_kpis")
    kpis = dict(cursor.fetchone())
    
    # Monthly sales
    cursor.execute("SELECT * FROM monthly_sales ORDER BY month")
    monthly = [dict(row) for row in cursor.fetchall()]
    
    # Weekday sales
    cursor.execute("SELECT * FROM weekday_sales")
    weekday = [dict(row) for row in cursor.fetchall()]
    
    # Payment methods
    cursor.execute("SELECT * FROM payment_methods ORDER BY revenue DESC")
    payments_dist = [dict(row) for row in cursor.fetchall()]
    
    # Order Status
    cursor.execute("SELECT * FROM order_status ORDER BY orders DESC")
    status_dist = [dict(row) for row in cursor.fetchall()]
    
    # State KPIs
    cursor.execute("SELECT * FROM state_kpis ORDER BY revenue DESC")
    state_dist = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "kpis": kpis,
        "monthly_trend": monthly,
        "weekday_trend": weekday,
        "payment_methods": payments_dist,
        "order_status": status_dist,
        "state_kpis": state_dist
    }

@app.get("/api/kpi/segmentation")
def get_segmentation_kpis():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM customer_segments_rfm ORDER BY revenue DESC")
    rfm = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM customer_segments_kmeans ORDER BY avg_monetary DESC")
    kmeans = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM customer_pca_scatter")
    pca = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "rfm_segments": rfm,
        "kmeans_segments": kmeans,
        "pca_scatter": pca
    }

@app.get("/api/kpi/logistics")
def get_logistics_kpis():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM logistics_delivery_distribution")
    delivery = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM logistics_rating_vs_delay ORDER BY delay_days")
    delay_ratings = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "delivery_distribution": delivery,
        "rating_vs_delay": delay_ratings
    }

@app.get("/api/kpi/marketplace")
def get_marketplace_kpis():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM seller_pareto ORDER BY seller_percentile")
    pareto = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM seller_rankings ORDER BY health_score DESC")
    rankings = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "seller_pareto": pareto,
        "seller_rankings": rankings
    }

@app.get("/api/kpi/products")
def get_product_kpis():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM product_categories ORDER BY revenue DESC")
    categories = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "product_categories": categories
    }

@app.get("/api/kpi/advanced")
def get_advanced_kpis():
    conn = get_db_conn()
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM clv_feature_importance ORDER BY importance DESC")
    importance = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM clv_predictions_scatter")
    scatter = [dict(row) for row in cursor.fetchall()]
    
    cursor.execute("SELECT * FROM revenue_forecast ORDER BY date")
    forecast = [dict(row) for row in cursor.fetchall()]
    
    conn.close()
    
    return {
        "clv_importance": importance,
        "clv_scatter": scatter,
        "revenue_forecast": forecast
    }

@app.get("/api/kpi/teleco/churn")
def get_teleco_churn():
    try:
        teleco_path = project_root / "datasets" / "teleco" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
        if not teleco_path.exists():
            raise HTTPException(status_code=404, detail="Teleco customer churn dataset not found.")
        
        # Load dataset
        df_teleco = pd.read_csv(teleco_path)
        total_customers = int(len(df_teleco))
        
        if total_customers == 0:
            return {
                "total_customers": 0,
                "churned_customers": 0,
                "retained_customers": 0,
                "churn_rate": 0.0,
                "retention_rate": 100.0,
                "avg_monthly_charges": 0.0,
                "avg_tenure": 0.0,
                "total_charges": 0.0,
                "contract_churn": {"labels": [], "rates": []},
                "payment_churn": {"labels": [], "rates": []},
                "internet_churn": {"labels": [], "rates": []},
                "monthly_charges_dist": {"labels": [], "retained": [], "churned": []},
                "tenure_dist": {"labels": [], "retained": [], "churned": []}
            }
            
        # Count Churn values
        churn_counts = df_teleco['Churn'].value_counts()
        churn_yes = int(churn_counts.get('Yes', 0))
        churn_no = int(churn_counts.get('No', 0))
        churn_rate = (churn_yes / total_customers) * 100
        
        # Compute other metrics
        avg_monthly_charges = float(df_teleco['MonthlyCharges'].mean())
        avg_tenure = float(df_teleco['tenure'].mean())
        
        # Parse and sum TotalCharges (dealing with empty strings)
        total_charges_series = df_teleco['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
        total_charges_series = pd.to_numeric(total_charges_series, errors='coerce').fillna(0)
        total_charges = float(total_charges_series.sum())
        
        # Calculate chart data
        # Contract Churn rates
        contract_rates = (df_teleco.groupby('Contract')['Churn'].value_counts(normalize=True).unstack(fill_value=0) * 100).round(2)
        contract_labels = contract_rates.index.tolist()
        contract_churn_rates = contract_rates['Yes'].tolist() if 'Yes' in contract_rates.columns else [0] * len(contract_labels)
        
        # Payment Churn rates
        payment_rates = (df_teleco.groupby('PaymentMethod')['Churn'].value_counts(normalize=True).unstack(fill_value=0) * 100).round(2)
        payment_labels = payment_rates.index.tolist()
        payment_churn_rates = payment_rates['Yes'].tolist() if 'Yes' in payment_rates.columns else [0] * len(payment_labels)
        
        # Internet Service Churn rates
        internet_rates = (df_teleco.groupby('InternetService')['Churn'].value_counts(normalize=True).unstack(fill_value=0) * 100).round(2)
        internet_labels = internet_rates.index.tolist()
        internet_churn_rates = internet_rates['Yes'].tolist() if 'Yes' in internet_rates.columns else [0] * len(internet_labels)
        
        # Binned Monthly Charges
        bins_monthly = np.linspace(15, 125, 12)
        df_teleco['MonthlyChargesBin'] = pd.cut(df_teleco['MonthlyCharges'], bins=bins_monthly)
        monthly_dist = df_teleco.groupby(['MonthlyChargesBin', 'Churn']).size().unstack(fill_value=0)
        monthly_labels = [f"{int(b.left)}-{int(b.right)}" for b in monthly_dist.index]
        monthly_retained = monthly_dist['No'].tolist() if 'No' in monthly_dist.columns else [0] * len(monthly_labels)
        monthly_churned = monthly_dist['Yes'].tolist() if 'Yes' in monthly_dist.columns else [0] * len(monthly_labels)
        
        # Binned Tenure
        bins_tenure = np.linspace(0, 72, 13)
        df_teleco['TenureBin'] = pd.cut(df_teleco['tenure'], bins=bins_tenure)
        tenure_dist = df_teleco.groupby(['TenureBin', 'Churn']).size().unstack(fill_value=0)
        tenure_labels = [f"{int(b.left)}-{int(b.right)}" for b in tenure_dist.index]
        tenure_retained = tenure_dist['No'].tolist() if 'No' in tenure_dist.columns else [0] * len(tenure_labels)
        tenure_churned = tenure_dist['Yes'].tolist() if 'Yes' in tenure_dist.columns else [0] * len(tenure_labels)
        
        return {
            "total_customers": total_customers,
            "churned_customers": churn_yes,
            "retained_customers": churn_no,
            "churn_rate": round(churn_rate, 2),
            "retention_rate": round(100.0 - churn_rate, 2),
            "avg_monthly_charges": round(avg_monthly_charges, 2),
            "avg_tenure": round(avg_tenure, 2),
            "total_charges": round(total_charges, 2),
            "contract_churn": {
                "labels": contract_labels,
                "rates": contract_churn_rates
            },
            "payment_churn": {
                "labels": payment_labels,
                "rates": payment_churn_rates
            },
            "internet_churn": {
                "labels": internet_labels,
                "rates": internet_churn_rates
            },
            "monthly_charges_dist": {
                "labels": monthly_labels,
                "retained": monthly_retained,
                "churned": monthly_churned
            },
            "tenure_dist": {
                "labels": tenure_labels,
                "retained": tenure_retained,
                "churned": tenure_churned
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate teleco churn rate: {str(e)}")

@app.post("/api/predict/teleco/churn")
def predict_teleco_churn(data: dict):
    try:
        global teleco_model, teleco_scaler, teleco_feature_names
        # Lazy load model if not loaded
        if teleco_model is None:
            teleco_model_path = project_root / "notebooks" / "teleco" / "final_churn_xgb_model.pkl"
            if not teleco_model_path.exists():
                raise HTTPException(status_code=404, detail="Teleco churn model not trained yet. Please run the model building notebook first.")
            teleco_payload = joblib.load(teleco_model_path)
            teleco_model = teleco_payload['model']
            teleco_scaler = teleco_payload['scaler']
            teleco_feature_names = teleco_payload['feature_names']
            
        # Parse inputs from request
        gender = 1 if data.get("gender") == "Female" else 0  # 1 for Female, 0 for Male
        senior_citizen = 1 if data.get("SeniorCitizen") == "Yes" else 0
        partner = 1 if data.get("Partner") == "Yes" else 0
        dependents = 1 if data.get("Dependents") == "Yes" else 0
        tenure = int(data.get("tenure", 1))
        phone_service = 1 if data.get("PhoneService") == "Yes" else 0
        paperless_billing = 1 if data.get("PaperlessBilling") == "Yes" else 0
        monthly_charges = float(data.get("MonthlyCharges", 50.0))
        total_charges = float(data.get("TotalCharges", monthly_charges * tenure))
        
        # Engineered features
        avg_monthly_spend = total_charges / (tenure + 1)
        
        services_count = 0
        services = ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
        for s in services:
            if data.get(s) == "Yes":
                services_count += 1
                
        # Initialize feature dictionary with zeroes
        feat_dict = {feat: 0 for feat in teleco_feature_names}
        
        # Set values for simple binary/numeric columns
        feat_dict['gender'] = gender
        feat_dict['SeniorCitizen'] = senior_citizen
        feat_dict['Partner'] = partner
        feat_dict['Dependents'] = dependents
        feat_dict['tenure'] = tenure
        feat_dict['PhoneService'] = phone_service
        feat_dict['PaperlessBilling'] = paperless_billing
        feat_dict['MonthlyCharges'] = monthly_charges
        feat_dict['TotalCharges'] = total_charges
        feat_dict['AvgMonthlySpend'] = avg_monthly_spend
        feat_dict['ServicesCount'] = services_count
        
        # TenureBucket
        if tenure > 48:
            feat_dict['TenureBucket_Loyal'] = 1
        elif tenure <= 12:
            feat_dict['TenureBucket_New'] = 1
            
        # MonthlyChargeBucket
        if monthly_charges > 70.35:
            feat_dict['MonthlyChargeBucket_High'] = 1
        elif monthly_charges > 29.0:
            feat_dict['MonthlyChargeBucket_Medium'] = 1
            
        # Multi-category mappings
        def map_multi(col_name, val):
            col_val_key = f"{col_name}_{val}"
            if col_val_key in feat_dict:
                feat_dict[col_val_key] = 1
                
        map_multi("MultipleLines", data.get("MultipleLines"))
        map_multi("InternetService", data.get("InternetService"))
        map_multi("OnlineSecurity", data.get("OnlineSecurity"))
        map_multi("OnlineBackup", data.get("OnlineBackup"))
        map_multi("DeviceProtection", data.get("DeviceProtection"))
        map_multi("TechSupport", data.get("TechSupport"))
        map_multi("StreamingTV", data.get("StreamingTV"))
        map_multi("StreamingMovies", data.get("StreamingMovies"))
        map_multi("Contract", data.get("Contract"))
        map_multi("PaymentMethod", data.get("PaymentMethod"))
        
        # Form input DataFrame for scaling & prediction
        df_input = pd.DataFrame([feat_dict], columns=teleco_feature_names)
        
        # Scale numeric features
        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'AvgMonthlySpend', 'ServicesCount']
        df_input[numerical_cols] = teleco_scaler.transform(df_input[numerical_cols])
        
        # Predict probability
        prob = float(teleco_model.predict_proba(df_input)[0, 1])
        prediction = int(teleco_model.predict(df_input)[0])
        
        # Risk factors highlights
        risk_factors = []
        if data.get("Contract") == "Month-to-month":
            risk_factors.append("Month-to-month Contract (High Risk)")
        if data.get("PaymentMethod") == "Electronic check":
            risk_factors.append("Payment via Electronic Check (High Risk)")
        if data.get("InternetService") == "Fiber optic":
            risk_factors.append("Fiber Optic Internet Subscription (High Risk)")
        if tenure <= 6:
            risk_factors.append("New Customer tenure <= 6 months (High Risk)")
        if monthly_charges >= 75:
            risk_factors.append("High Monthly Charges >= $75 (High Risk)")
            
        mitigators = []
        if data.get("Contract") in ["One year", "Two year"]:
            mitigators.append("Long-term Contract (Protective)")
        if tenure > 24:
            mitigators.append("Established customer tenure > 2 years (Protective)")
        if services_count >= 3:
            mitigators.append(f"Multiple active services: {services_count} (Protective)")
            
        return {
            "status": "success",
            "churn_probability": round(prob * 100, 2),
            "prediction": "Churn" if prediction == 1 or prob >= 0.5 else "Retained",
            "risk_factors": risk_factors[:3],
            "mitigating_factors": mitigators[:3]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# INSTACART ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/kpi/instacart/overview")
def get_instacart_overview():
    """Headline KPIs + hour/weekday order trends + top products + department sales."""
    try:
        data = get_instacart_data()
        if data is None:
            raise HTTPException(status_code=404, detail="Instacart dataset not found.")

        orders = data["orders"]
        df     = data["df"]

        total_orders          = int(orders["order_id"].nunique())
        unique_customers      = int(orders["user_id"].nunique())
        unique_products       = int(df["product_id"].nunique())
        num_departments       = int(df["department"].nunique())
        basket_sizes          = df.groupby("order_id")["product_id"].count()
        avg_basket_size       = float(basket_sizes.mean())
        repeat_purchase_rate  = float(df["reordered"].mean() * 100)
        avg_orders_per_customer = float(orders.groupby("user_id")["order_id"].nunique().mean())

        # Hour-of-day distribution
        hour_orders = orders["order_hour_of_day"].value_counts().sort_index()

        # Day-of-week distribution
        dow_raw    = orders["order_dow"].value_counts().sort_index()
        dow_labels = ["Sunday","Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"]

        # Top 15 products
        top_15 = df["product_name"].value_counts().head(15)

        # Department sales
        dept_sales = df["department"].value_counts()

        return {
            "total_orders":           total_orders,
            "unique_customers":       unique_customers,
            "unique_products":        unique_products,
            "num_departments":        num_departments,
            "avg_basket_size":        round(avg_basket_size, 2),
            "repeat_purchase_rate":   round(repeat_purchase_rate, 2),
            "avg_orders_per_customer": round(avg_orders_per_customer, 2),
            "hour_orders": {
                "labels": hour_orders.index.tolist(),
                "values": hour_orders.values.tolist()
            },
            "dow_orders": {
                "labels": [dow_labels[i] for i in dow_raw.index.tolist()],
                "values": dow_raw.values.tolist()
            },
            "top_15_products": {
                "names":  top_15.index.tolist(),
                "counts": top_15.values.tolist()
            },
            "dept_sales": {
                "labels": dept_sales.index.tolist(),
                "values": dept_sales.values.tolist()
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kpi/instacart/products")
def get_instacart_products():
    """Top-20 products with reorder rates, department breakdown, top aisles."""
    try:
        data = get_instacart_data()
        if data is None:
            raise HTTPException(status_code=404, detail="Instacart dataset not found.")

        df = data["df"]

        prod_stats = (
            df.groupby("product_name")
            .agg(total_purchases=("order_id", "count"), reorder_rate=("reordered", "mean"))
            .reset_index()
            .sort_values("total_purchases", ascending=False)
            .head(20)
        )
        prod_stats["reorder_rate"] = prod_stats["reorder_rate"].round(4)

        dept_breakdown = (
            df.groupby("department")
            .agg(total_purchases=("order_id", "count"), unique_products=("product_id", "nunique"))
            .reset_index()
            .sort_values("total_purchases", ascending=False)
        )

        top_aisles = (
            df.groupby(["department", "aisle"]).size()
            .reset_index(name="total_purchases")
            .sort_values("total_purchases", ascending=False)
            .head(30)
        )

        return {
            "top_20_products": prod_stats.to_dict("records"),
            "dept_breakdown":  dept_breakdown.to_dict("records"),
            "top_aisles":      top_aisles.to_dict("records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/kpi/instacart/association")
def get_instacart_association():
    """Market basket association rules (co-occurrence on top-200 products) + customer segment stats."""
    try:
        from itertools import combinations
        from collections import Counter

        data = get_instacart_data()
        if data is None:
            raise HTTPException(status_code=404, detail="Instacart dataset not found.")

        df     = data["df"]
        orders = data["orders"]

        # Restrict to top-200 products for speed
        top_products = df["product_name"].value_counts().head(200).index
        df_f = df[df["product_name"].isin(top_products)]
        n_orders = int(df_f["order_id"].nunique())

        prod_counts  = df_f.groupby("product_name")["order_id"].nunique()
        prod_support = (prod_counts / n_orders).to_dict()

        order_products = df_f.groupby("order_id")["product_name"].apply(set)

        pair_counts: Counter = Counter()
        for prods in order_products:
            lst = sorted(list(prods))
            if len(lst) > 1:
                pair_counts.update(combinations(lst, 2))

        rules = []
        for (ant, cons), count in pair_counts.most_common(500):
            support    = count / n_orders
            if support < 0.005:
                continue
            confidence = count / max(prod_counts.get(ant, 1), 1)
            ant_s      = prod_support.get(ant, 1e-9)
            cons_s     = prod_support.get(cons, 1e-9)
            lift       = support / (ant_s * cons_s) if ant_s * cons_s > 0 else 0.0
            rules.append({
                "antecedent":  ant,
                "consequent":  cons,
                "support":     round(support, 4),
                "confidence":  round(confidence, 4),
                "lift":        round(lift, 4)
            })
        rules.sort(key=lambda x: x["lift"], reverse=True)
        rules = rules[:20]

        # Simple rule-based customer segment counts
        cust_feat = df.groupby("user_id").agg(
            total_orders=("order_number", "max"),
            reorder_rate=("reordered", "mean")
        ).reset_index()

        heavy_count    = int((cust_feat["total_orders"] >= 10).sum())
        occasional_count = int((cust_feat["total_orders"] <= 3).sum())

        df2 = df.copy()
        df2["is_organic"] = df2["product_name"].str.contains("Organic", case=False)
        organic_share = df2.groupby("user_id")["is_organic"].mean()
        organic_count = int((organic_share >= 0.30).sum())

        segment_stats = [
            {"segment": "Heavy Buyers",     "count": heavy_count,    "description": "Customers with ≥10 prior orders",         "color": "#6366f1"},
            {"segment": "Organic Shoppers", "count": organic_count,  "description": "Customers with ≥30% organic products",    "color": "#10b981"},
            {"segment": "Occasional Buyers","count": occasional_count,"description": "Customers with ≤3 prior orders",         "color": "#f59e0b"},
        ]

        return {
            "rules":         rules,
            "segment_stats": segment_stats,
            "total_orders":  n_orders
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/kpi/recommendations")
def get_recommendations(user_id: Optional[str] = None):
    conn = get_db_conn()
    cursor = conn.cursor()
    
    # 1. Fetch sample active customers for dropdown UI selection
    cursor.execute("SELECT * FROM active_customers_sample")
    active_samples = [dict(row) for row in cursor.fetchall()]
    
    if not user_id:
        conn.close()
        return {
            "active_samples": active_samples,
            "purchase_history": [],
            "svd_recommendations": [],
            "similarity_recommendations": []
        }
        
    # 2. Fetch purchase history of specified user
    cursor.execute(
        "SELECT * FROM customer_purchase_history WHERE customer_unique_id = ? ORDER BY purchase_date DESC",
        (user_id,)
    )
    history = [dict(row) for row in cursor.fetchall()]
    
    # 3. Dynamic SVD Recommendations (using trained surprise SVD model)
    svd_recs = []
    if svd_model is not None and candidate_products:
        already_purchased = set(item["product_id"] for item in history)
        
        preds = []
        for pid in candidate_products:
            if pid not in already_purchased:
                est_rating = svd_model.predict(user_id, pid).est
                preds.append((pid, est_rating))
                
        # Sort and take top 10
        preds.sort(key=lambda x: x[1], reverse=True)
        
        # Add metadata for recommended products (like price, category from DB)
        top_preds = preds[:10]
        if top_preds:
            placeholders = ",".join("?" for _ in top_preds)
            cursor.execute(
                f"SELECT product_id, product_category, price FROM customer_purchase_history WHERE product_id IN ({placeholders}) GROUP BY product_id",
                [p[0] for p in top_preds]
            )
            meta_map = {row["product_id"]: (row["product_category"], row["price"]) for row in cursor.fetchall()}
            
            for pid, score in top_preds:
                cat, price = meta_map.get(pid, ("unknown", 0.0))
                svd_recs.append({
                    "product_id": pid,
                    "predicted_rating": round(score, 3),
                    "product_category": cat,
                    "price": price
                })
                
    # 4. Item-Based Similarity Recommendations (similar to user's last purchased item)
    similarity_recs = []
    if history:
        last_purchased_prod = history[0]["product_id"]
        cursor.execute(
            "SELECT * FROM product_similarity_top WHERE product_id = ? ORDER BY similarity DESC LIMIT 10",
            (last_purchased_prod,)
        )
        sim_rows = cursor.fetchall()
        
        if sim_rows:
            sim_pids = [row["similar_product_id"] for row in sim_rows]
            placeholders = ",".join("?" for _ in sim_pids)
            cursor.execute(
                f"SELECT product_id, product_category, price FROM customer_purchase_history WHERE product_id IN ({placeholders}) GROUP BY product_id",
                sim_pids
            )
            meta_map = {row["product_id"]: (row["product_category"], row["price"]) for row in cursor.fetchall()}
            
            for row in sim_rows:
                sim_pid = row["similar_product_id"]
                cat, price = meta_map.get(sim_pid, ("unknown", 0.0))
                similarity_recs.append({
                    "product_id": sim_pid,
                    "similarity": round(row["similarity"], 3),
                    "product_category": cat,
                    "price": price,
                    "reference_product_id": last_purchased_prod
                })
                
    conn.close()
    
    return {
        "active_samples": active_samples,
        "purchase_history": history,
        "svd_recommendations": svd_recs,
        "similarity_recommendations": similarity_recs
    }

# Explicit no-cache route for index.html
@app.get("/")
async def serve_index():
    index_path = frontend_dir / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Frontend not found")
    return FileResponse(
        str(index_path),
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

# Serve all other static assets (JS, CSS, etc.) — no-cache
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=False), name="static")
else:
    print(f"Frontend static directory not found at {frontend_dir}. API will run in API-only mode.")
