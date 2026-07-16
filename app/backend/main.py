import sqlite3
import pickle
import warnings
import sys
import subprocess
from pathlib import Path
from typing import Optional, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

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

@app.on_event("startup")
def load_assets():
    global svd_model, candidate_products
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

# Serve frontend static assets if directory exists
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="static")
else:
    print(f"Frontend static directory not found at {frontend_dir}. API will run in API-only mode.")
