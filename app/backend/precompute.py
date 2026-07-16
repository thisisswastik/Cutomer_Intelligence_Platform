import os
import sqlite3
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

# Ignore warnings
warnings.filterwarnings("ignore")

# Machine Learning imports
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from prophet import Prophet
from surprise import Dataset, Reader, SVD

def main():
    # Define paths relative to this file
    backend_dir = Path(__file__).resolve().parent
    project_root = backend_dir.parents[1]
    data_dir = project_root / "datasets" / "Olist"
    output_dir = project_root / "datasets" / "precomputed"
    output_dir.mkdir(parents=True, exist_ok=True)
    db_path = output_dir / "olist_dashboard.db"
    
    print(f"Project root: {project_root}")
    print(f"Data directory: {data_dir}")
    print(f"Output database path: {db_path}")
    
    # Check if database already exists
    if db_path.exists():
        print("Removing existing precomputed database...")
        db_path.unlink()
        
    conn = sqlite3.connect(str(db_path))
    
    # -------------------------------------------------------------
    # 1. Load Raw Datasets
    # -------------------------------------------------------------
    print("\n[1/9] Loading raw Olist datasets...")
    orders = pd.read_csv(
        data_dir / "olist_orders_dataset.csv",
        parse_dates=[
            "order_purchase_timestamp",
            "order_approved_at",
            "order_delivered_carrier_date",
            "order_delivered_customer_date",
            "order_estimated_delivery_date"
        ]
    )
    payments = pd.read_csv(data_dir / "olist_order_payments_dataset.csv")
    customers = pd.read_csv(data_dir / "olist_customers_dataset.csv")
    order_items = pd.read_csv(data_dir / "olist_order_items_dataset.csv")
    reviews = pd.read_csv(data_dir / "olist_order_reviews_dataset.csv")
    products = pd.read_csv(data_dir / "olist_products_dataset.csv")
    sellers = pd.read_csv(data_dir / "olist_sellers_dataset.csv")
    translation = pd.read_csv(data_dir / "product_category_name_translation.csv")
    
    # -------------------------------------------------------------
    # 2. Executive KPIs & Trends
    # -------------------------------------------------------------
    print("\n[2/9] Computing Executive KPIs...")
    # Master df for executive KPI (Orders + Payments + Customers + Reviews)
    master_df = orders.merge(payments, on="order_id", how="left")
    master_df = master_df.merge(customers, on="customer_id", how="left")
    
    total_revenue = float(payments["payment_value"].sum())
    total_orders = int(orders["order_id"].nunique())
    total_customers = int(customers["customer_unique_id"].nunique())
    aov = total_revenue / total_orders
    avg_review = float(reviews["review_score"].mean())
    
    # Delivery times
    delivery_days = (orders["order_delivered_customer_date"] - orders["order_purchase_timestamp"]).dt.days
    avg_delivery_days = float(delivery_days.mean())
    
    # Delayed orders pct
    valid_deliveries = orders.dropna(subset=["order_delivered_customer_date", "order_estimated_delivery_date"])
    delayed_orders_count = int((valid_deliveries["order_delivered_customer_date"] > valid_deliveries["order_estimated_delivery_date"]).sum())
    delayed_pct = (delayed_orders_count / len(valid_deliveries)) * 100 if len(valid_deliveries) > 0 else 0.0
    
    # Avg delay days (for delayed orders)
    delays = (valid_deliveries["order_delivered_customer_date"] - valid_deliveries["order_estimated_delivery_date"]).dt.days
    avg_delay_days = float(delays[delays > 0].mean()) if (delays > 0).sum() > 0 else 0.0
    
    # Write Executive Totals
    exec_kpis = pd.DataFrame([{
        "total_revenue": total_revenue,
        "total_orders": total_orders,
        "total_customers": total_customers,
        "aov": aov,
        "avg_review": avg_review,
        "avg_delivery_days": avg_delivery_days,
        "delayed_order_pct": delayed_pct,
        "avg_delay_days": avg_delay_days
    }])
    exec_kpis.to_sql("executive_kpis", conn, index=False, if_exists="replace")
    
    # Monthly sales trend
    monthly_sales = master_df.copy()
    monthly_sales["month"] = monthly_sales["order_purchase_timestamp"].dt.to_period("M").astype(str)
    monthly_trend = (
        monthly_sales
        .groupby("month")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "nunique")
        )
        .reset_index()
        .sort_values("month")
    )
    monthly_trend.to_sql("monthly_sales", conn, index=False, if_exists="replace")
    
    # Weekday sales trend
    master_df["Weekday"] = master_df["order_purchase_timestamp"].dt.day_name()
    weekday_trend = (
        master_df
        .groupby("Weekday")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "nunique")
        )
        .reindex(["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        .reset_index()
    )
    weekday_trend.to_sql("weekday_sales", conn, index=False, if_exists="replace")
    
    # Payment method share
    payment_share = (
        payments
        .groupby("payment_type")
        .agg(
            revenue=("payment_value", "sum"),
            orders=("order_id", "nunique")
        )
        .reset_index()
        .sort_values("revenue", ascending=False)
    )
    payment_share.to_sql("payment_methods", conn, index=False, if_exists="replace")
    
    # Order Status Distribution
    order_status_dist = (
        orders
        .groupby("order_status")
        .agg(orders=("order_id", "nunique"))
        .reset_index()
        .sort_values("orders", ascending=False)
    )
    order_status_dist.to_sql("order_status", conn, index=False, if_exists="replace")
    
    # State-wise KPIs
    state_kpis = (
        master_df
        .groupby("customer_state")
        .agg(
            revenue=("payment_value", "sum"),
            customers=("customer_unique_id", "nunique"),
            orders=("order_id", "nunique")
        )
        .reset_index()
    )
    # Add average delivery days and average delay by state
    orders_cust = orders.merge(customers, on="customer_id", how="left")
    orders_cust["delivery_days"] = (orders_cust["order_delivered_customer_date"] - orders_cust["order_purchase_timestamp"]).dt.days
    orders_cust["delay_days"] = (orders_cust["order_delivered_customer_date"] - orders_cust["order_estimated_delivery_date"]).dt.days
    
    state_delivery = (
        orders_cust
        .groupby("customer_state")
        .agg(
            avg_delivery_days=("delivery_days", "mean"),
            avg_delay_days=("delay_days", lambda x: x[x > 0].mean() if (x > 0).sum() > 0 else 0.0)
        )
        .reset_index()
    )
    state_kpis = state_kpis.merge(state_delivery, on="customer_state", how="left").fillna(0.0)
    state_kpis.to_sql("state_kpis", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 3. Customer Segmentation (RFM & KMeans)
    # -------------------------------------------------------------
    print("\n[3/9] Performing Customer Segmentation...")
    customer_df = customers.merge(orders, on="customer_id").merge(payments, on="order_id")
    snapshot_date = customer_df["order_purchase_timestamp"].max() + pd.Timedelta(days=1)
    
    rfm = (
        customer_df
        .groupby("customer_unique_id")
        .agg({
            "order_purchase_timestamp": "max",
            "order_id": "nunique",
            "payment_value": "sum"
        })
    )
    rfm.columns = ["LastPurchase", "Frequency", "Monetary"]
    rfm["Recency"] = (snapshot_date - rfm["LastPurchase"]).dt.days
    rfm = rfm[["Recency", "Frequency", "Monetary"]]
    
    # Scores
    rfm["R"] = pd.qcut(rfm["Recency"], 5, labels=[5, 4, 3, 2, 1])
    rfm["F"] = pd.qcut(rfm["Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
    rfm["M"] = pd.qcut(rfm["Monetary"], 5, labels=[1, 2, 3, 4, 5])
    
    def get_segment(row):
        if row["R"] >= 4 and row["F"] >= 4:
            return "Champions"
        elif row["R"] >= 3 and row["F"] >= 3:
            return "Loyal Customer"
        elif row["R"] >= 4:
            return "Potential Loyalists"
        elif row["R"] <= 2 and row["F"] >= 4:
            return "At Risk"
        else:
            return "Others"
            
    rfm["Segment"] = rfm.apply(get_segment, axis=1)
    
    segment_kpi = (
        rfm
        .groupby("Segment")
        .agg(
            customers=("Monetary", "count"),
            revenue=("Monetary", "sum"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean")
        )
        .reset_index()
    )
    segment_kpi.to_sql("customer_segments_rfm", conn, index=False, if_exists="replace")
    
    # KMeans Clustering
    kmeans_features = (
        customer_df
        .groupby("customer_unique_id")
        .agg(
            Recency=("order_purchase_timestamp", lambda x: (snapshot_date - x.max()).days),
            Frequency=("order_id", "nunique"),
            Monetary=("payment_value", "sum"),
            AvgOrderValue=("payment_value", "mean")
        )
    )
    
    features_log = kmeans_features.copy()
    features_log["Monetary"] = np.log1p(features_log["Monetary"])
    features_log["Frequency"] = np.log1p(features_log["Frequency"])
    features_log["AvgOrderValue"] = np.log1p(features_log["AvgOrderValue"])
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_log)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    kmeans_features["Cluster"] = kmeans.fit_predict(features_scaled)
    
    cluster_profile = (
        kmeans_features
        .groupby("Cluster")
        .agg(
            customers=("Monetary", "count"),
            avg_recency=("Recency", "mean"),
            avg_frequency=("Frequency", "mean"),
            avg_monetary=("Monetary", "mean"),
            avg_order_value=("AvgOrderValue", "mean")
        )
        .reset_index()
    )
    cluster_profile.to_sql("customer_segments_kmeans", conn, index=False, if_exists="replace")
    
    # PCA
    pca = PCA(n_components=2)
    pca_components = pca.fit_transform(features_scaled)
    pca_df = pd.DataFrame({
        "customer_unique_id": kmeans_features.index,
        "pc1": pca_components[:, 0],
        "pc2": pca_components[:, 1],
        "cluster": kmeans_features["Cluster"].values
    })
    pca_sample = pca_df.sample(n=min(2500, len(pca_df)), random_state=42)
    pca_sample.to_sql("customer_pca_scatter", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 4. Logistics & Delivery Performance
    # -------------------------------------------------------------
    print("\n[4/9] Calculating Logistics & Delivery statistics...")
    valid_orders_delivery = orders.dropna(subset=["order_delivered_customer_date", "order_purchase_timestamp"])
    delivery_days_all = (valid_orders_delivery["order_delivered_customer_date"] - valid_orders_delivery["order_purchase_timestamp"]).dt.days
    
    def bucket_days(d):
        if d <= 5: return "0-5 days"
        elif d <= 10: return "6-10 days"
        elif d <= 15: return "11-15 days"
        elif d <= 20: return "16-20 days"
        elif d <= 25: return "21-25 days"
        elif d <= 30: return "26-30 days"
        elif d <= 45: return "31-45 days"
        else: return "45+ days"
        
    delivery_buckets = delivery_days_all.apply(bucket_days).value_counts().reset_index()
    delivery_buckets.columns = ["days_bucket", "order_count"]
    bucket_order = ["0-5 days", "6-10 days", "11-15 days", "16-20 days", "21-25 days", "26-30 days", "31-45 days", "45+ days"]
    delivery_buckets["days_bucket"] = pd.Categorical(delivery_buckets["days_bucket"], categories=bucket_order, ordered=True)
    delivery_buckets = delivery_buckets.sort_values("days_bucket")
    delivery_buckets.to_sql("logistics_delivery_distribution", conn, index=False, if_exists="replace")
    
    # Rating vs delivery delay
    orders_reviews = orders.merge(reviews, on="order_id", how="inner")
    orders_reviews["delay_days"] = (orders_reviews["order_delivered_customer_date"] - orders_reviews["order_estimated_delivery_date"]).dt.days
    delayed_reviews = orders_reviews[orders_reviews["delay_days"] > 0]
    rating_vs_delay = (
        delayed_reviews
        .groupby("delay_days")
        .agg(
            avg_review_score=("review_score", "mean"),
            order_count=("order_id", "nunique")
        )
        .reset_index()
        .query("delay_days <= 45")
    )
    rating_vs_delay.to_sql("logistics_rating_vs_delay", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 5. Marketplace & Seller Performance
    # -------------------------------------------------------------
    print("\n[5/9] Calculating Seller Pareto & rankings...")
    seller_df = order_items.merge(orders, on="order_id", how="left")
    seller_df = seller_df.merge(reviews, on="order_id", how="left")
    seller_df = seller_df.merge(sellers, on="seller_id", how="left")
    
    seller_scores = (
        seller_df
        .groupby("seller_id")
        .agg(
            revenue=("price", "sum"),
            orders=("order_id", "nunique"),
            rating=("review_score", "mean"),
            state=("seller_state", "first")
        )
        .fillna(0.0)
    )
    
    # Pareto Analysis
    seller_pareto_df = seller_scores.sort_values("revenue", ascending=False).copy()
    seller_pareto_df["cum_revenue"] = seller_pareto_df["revenue"].cumsum()
    total_seller_revenue = seller_pareto_df["revenue"].sum()
    seller_pareto_df["revenue_share"] = (seller_pareto_df["cum_revenue"] / total_seller_revenue) * 100
    seller_pareto_df["seller_rank"] = np.arange(1, len(seller_pareto_df) + 1)
    seller_pareto_df["seller_percentile"] = (seller_pareto_df["seller_rank"] / len(seller_pareto_df)) * 100
    
    pareto_curve = (
        seller_pareto_df[["seller_percentile", "revenue_share"]]
        .round(2)
        .iloc[::max(1, len(seller_pareto_df)//100)]
        .reset_index(drop=True)
    )
    pareto_curve.to_sql("seller_pareto", conn, index=False, if_exists="replace")
    
    # Seller Score
    max_rev = seller_scores["revenue"].max()
    max_ord = seller_scores["orders"].max()
    seller_scores["health_score"] = (
        0.5 * (seller_scores["revenue"] / max_rev)
        + 0.3 * (seller_scores["orders"] / max_ord)
        + 0.2 * (seller_scores["rating"] / 5.0)
    ) * 100
    
    top_sellers = (
        seller_scores
        .sort_values("health_score", ascending=False)
        .head(100)
        .reset_index()
    )
    top_sellers.to_sql("seller_rankings", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 6. Product Sales & Category analysis
    # -------------------------------------------------------------
    print("\n[6/9] Analyzing Product Categories...")
    products_merged = order_items.merge(products, on="product_id", how="left").merge(translation, on="product_category_name", how="left")
    product_cats = (
        products_merged
        .groupby("product_category_name_english")
        .agg(
            revenue=("price", "sum"),
            orders=("order_id", "nunique"),
            avg_price=("price", "mean"),
            avg_freight=("freight_value", "mean")
        )
        .reset_index()
        .rename(columns={"product_category_name_english": "category"})
        .sort_values("revenue", ascending=False)
    )
    product_cats.to_sql("product_categories", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 7. Customer Lifetime Value (CLV)
    # -------------------------------------------------------------
    print("\n[7/9] Training Random Forest for CLV prediction...")
    ml_dataset = pd.read_csv(data_dir / "customer_ml_dataset.csv", index_col=0)
    X = ml_dataset.drop(columns=["Monetary", "TotalPayment", "AvgPayment", "AvgOrderValue"], errors="ignore")
    y = ml_dataset["Monetary"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    
    clv_imp = pd.DataFrame({
        "feature": X.columns,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False)
    clv_imp.to_sql("clv_feature_importance", conn, index=False, if_exists="replace")
    
    clv_scatter = pd.DataFrame({
        "actual": y_test,
        "predicted": rf_pred
    }).sample(n=min(1000, len(y_test)), random_state=42)
    clv_scatter.to_sql("clv_predictions_scatter", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 8. Sales Forecasting (Prophet)
    # -------------------------------------------------------------
    print("\n[8/9] Fitting Prophet time-series sales forecasting model...")
    daily_sales = master_df.groupby(master_df["order_purchase_timestamp"].dt.date)["payment_value"].sum().reset_index()
    daily_sales.columns = ["Date", "Revenue"]
    daily_sales["Date"] = pd.to_datetime(daily_sales["Date"])
    daily_sales = daily_sales[daily_sales["Revenue"] > 100]
    
    prophet_df = daily_sales[["Date", "Revenue"]].copy()
    prophet_df.columns = ["ds", "y"]
    
    model_prophet = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model_prophet.fit(prophet_df)
    
    future = model_prophet.make_future_dataframe(periods=30)
    forecast = model_prophet.predict(future)
    
    forecast_results = forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    forecast_results.columns = ["date", "predicted", "lower_bound", "upper_bound"]
    forecast_results["date"] = pd.to_datetime(forecast_results["date"])
    
    daily_sales_join = daily_sales.copy()
    daily_sales_join.columns = ["date", "actual"]
    daily_sales_join["date"] = pd.to_datetime(daily_sales_join["date"])
    
    forecast_final = pd.merge(forecast_results, daily_sales_join, on="date", how="left")
    forecast_final["date"] = forecast_final["date"].dt.strftime("%Y-%m-%d")
    forecast_final.to_sql("revenue_forecast", conn, index=False, if_exists="replace")
    
    # -------------------------------------------------------------
    # 9. Recommendation System (SVD & Similarity)
    # -------------------------------------------------------------
    print("\n[9/9] Training SVD recommendation model & similarity index...")
    recommend_df = customers.merge(orders, on="customer_id").merge(order_items, on="order_id").merge(products, on="product_id")
    recommend_df = recommend_df.merge(translation, on="product_category_name", how="left")
    recommend_df = recommend_df[recommend_df["order_status"] == "delivered"]
    
    customer_activity = recommend_df.groupby("customer_unique_id").agg(Purchases=("order_id", "nunique"))
    popular_products = recommend_df.groupby("product_id").size()
    
    active_users = customer_activity[customer_activity["Purchases"] >= 2].index
    popular_products_idx = popular_products[popular_products >= 5].index
    
    if len(active_users) < 100:
        active_users = customer_activity[customer_activity["Purchases"] >= 1].index
    if len(popular_products_idx) < 100:
        popular_products_idx = popular_products[popular_products >= 2].index
        
    recommend_df_filtered = recommend_df[
        recommend_df["customer_unique_id"].isin(active_users) &
        recommend_df["product_id"].isin(popular_products_idx)
    ]
    
    user_item = recommend_df_filtered.pivot_table(
        index="customer_unique_id",
        columns="product_id",
        values="order_id",
        aggfunc="count",
        fill_value=0
    )
    user_item = (user_item > 0).astype(int)
    
    print("Computing product cosine similarity matrix...")
    prod_sim = cosine_similarity(user_item.T)
    prod_sim_df = pd.DataFrame(prod_sim, index=user_item.columns, columns=user_item.columns)
    
    top_similarities = []
    for pid in prod_sim_df.index:
        sims = prod_sim_df[pid].sort_values(ascending=False).iloc[1:11]
        for rank, (sim_pid, score) in enumerate(sims.items(), 1):
            if score > 0.05:
                top_similarities.append({
                    "product_id": pid,
                    "similar_product_id": sim_pid,
                    "similarity": float(score),
                    "rank_idx": rank
                })
    top_sim_df = pd.DataFrame(top_similarities)
    top_sim_df.to_sql("product_similarity_top", conn, index=False, if_exists="replace")
    
    print("Training Surprise SVD model...")
    interaction = recommend_df_filtered.groupby(["customer_unique_id", "product_id"]).size().reset_index(name="rating")
    reader = Reader(rating_scale=(1, max(2, interaction["rating"].max())))
    data = Dataset.load_from_df(interaction[["customer_unique_id", "product_id", "rating"]], reader)
    trainset = data.build_full_trainset()
    
    svd = SVD(random_state=42)
    svd.fit(trainset)
    
    model_pickle_path = output_dir / "svd_model.pkl"
    with open(model_pickle_path, "wb") as f:
        pickle.dump(svd, f)
    print(f"SVD recommendation model saved to {model_pickle_path}")
    
    print("Saving customer purchase history...")
    purchase_history = (
        recommend_df
        .groupby(["customer_unique_id", "order_id", "product_id"])
        .agg(
            product_category=("product_category_name_english", "first"),
            price=("price", "first"),
            purchase_date=("order_purchase_timestamp", lambda x: x.max().strftime("%Y-%m-%d"))
        )
        .reset_index()
    )
    purchase_history.to_sql("customer_purchase_history", conn, index=False, if_exists="replace")
    
    print("Saving active customers sample list...")
    active_cust_list = (
        rfm[rfm["Frequency"] >= 2]
        .reset_index()
        .merge(customers[["customer_unique_id", "customer_city", "customer_state"]].drop_duplicates("customer_unique_id"), on="customer_unique_id")
        .head(100)
    )
    if len(active_cust_list) < 20:
        active_cust_list = (
            rfm.reset_index()
            .merge(customers[["customer_unique_id", "customer_city", "customer_state"]].drop_duplicates("customer_unique_id"), on="customer_unique_id")
            .head(100)
        )
    active_cust_list.to_sql("active_customers_sample", conn, index=False, if_exists="replace")
    
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_hist_cust ON customer_purchase_history (customer_unique_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sim_prod ON product_similarity_top (product_id);")
    conn.commit()
    
    conn.close()
    print("\nPrecomputation complete! Database successfully generated and indexed.")

if __name__ == "__main__":
    main()
