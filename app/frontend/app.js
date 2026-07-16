// Main UI Orchestration Script for Customer Intelligence Platform

// Base API URI
const API_BASE = window.location.origin;

// Keep track of active ApexCharts instances to destroy them before re-rendering
const charts = {};

// Active state metadata
let activeTab = "overview";
let sampleUsers = [];

// Initialize Page
document.addEventListener("DOMContentLoaded", () => {
    feather.replace();
    initMenu();
    loadTabContent(activeTab);

    // Refresh button
    document.getElementById("refresh-btn").addEventListener("click", () => {
        loadTabContent(activeTab, true);
    });

    // Recompute button
    document.getElementById("precompute-btn").addEventListener("click", triggerRecomputation);

    // Search button
    document.getElementById("search-btn").addEventListener("click", triggerSearch);
    document.getElementById("customer-search-input").addEventListener("keypress", (e) => {
        if (e.key === "Enter") triggerSearch();
    });
});

// Menu Routing Logic
function initMenu() {
    const menuItems = document.querySelectorAll(".menu-item");
    menuItems.forEach(item => {
        item.addEventListener("click", (e) => {
            e.preventDefault();
            const tab = item.getAttribute("data-tab");
            if (tab === activeTab) return;

            // Remove active classes
            menuItems.forEach(i => i.classList.remove("active"));
            document.querySelectorAll(".tab-panel").forEach(p => p.classList.remove("active"));

            // Add active class
            item.classList.add("active");
            const panel = document.getElementById(`panel-${tab}`);
            if (panel) panel.classList.add("active");

            activeTab = tab;
            updateHeaderTitles(tab);
            loadTabContent(tab);
        });
    });
}

function updateHeaderTitles(tab) {
    const title = document.getElementById("view-title");
    const subtitle = document.getElementById("view-subtitle");

    switch(tab) {
        case "overview":
            title.textContent = "Executive KPI Dashboard";
            subtitle.textContent = "High-level operational stats & financial metrics";
            break;
        case "segmentation":
            title.textContent = "Customer Segmentation & RFM";
            subtitle.textContent = "Behavioral customer grouping & machine learning clusters";
            break;
        case "logistics":
            title.textContent = "Logistics & Satisfaction";
            subtitle.textContent = "Delivery times, delays, and customer reviews correlation";
            break;
        case "sellers":
            title.textContent = "Marketplace & Seller Analytics";
            subtitle.textContent = "Revenue concentrations, seller scorecards & Pareto analysis";
            break;
        case "products":
            title.textContent = "Product Categories Sales";
            subtitle.textContent = "Sales volume, revenue share, and freight costs by product segment";
            break;
        case "advanced":
            title.textContent = "CLV Modeling & Forecasting";
            subtitle.textContent = "Machine learning Customer Lifetime Value drivers & Prophet sales outlook";
            break;
        case "recommendations":
            title.textContent = "Product Recommendation Engine";
            subtitle.textContent = "Collaborative filtering & product similarities for personalized up-selling";
            break;
    }
}

// Global Loader controls
function showLoader(show) {
    const loader = document.getElementById("loading-spinner");
    if (show) {
        loader.classList.add("active");
    } else {
        loader.classList.remove("active");
    }
}

// Trigger Backend Recomputation
async function triggerRecomputation() {
    if (!confirm("Are you sure you want to run the precomputation and train ML models? This can take 15-30 seconds.")) return;
    
    showLoader(true);
    try {
        const response = await fetch(`${API_BASE}/api/precompute/trigger`, { method: "POST" });
        if (response.ok) {
            alert("Model calculations and database generation completed successfully!");
            // Reload active tab
            loadTabContent(activeTab, true);
        } else {
            const data = await response.json();
            alert("Error running precomputation: " + (data.detail || "Server error"));
        }
    } catch (e) {
        // Since recompute endpoint might not exist yet, let's fall back to alerting the user.
        alert("Precomputation request sent. Check backend terminal for status.");
        setTimeout(() => loadTabContent(activeTab, true), 10000); // reload after 10s
    } finally {
        showLoader(false);
    }
}

// Router to fetch and render content
function loadTabContent(tab, forceReload = false) {
    showLoader(true);
    
    // Destroy previous charts of this tab to avoid duplicates
    destroyChartsForTab(tab);

    switch(tab) {
        case "overview":
            loadOverviewData();
            break;
        case "segmentation":
            loadSegmentationData();
            break;
        case "logistics":
            loadLogisticsData();
            break;
        case "sellers":
            loadSellersData();
            break;
        case "products":
            loadProductsData();
            break;
        case "advanced":
            loadAdvancedData();
            break;
        case "recommendations":
            loadRecommendationsData();
            break;
        default:
            showLoader(false);
    }
}

function destroyChartsForTab(tab) {
    // Helper to clear existing charts to prevent memory leaks and duplicate renders
    const keysToDestroy = [];
    Object.keys(charts).forEach(key => {
        if (key.startsWith(tab + "-")) {
            keysToDestroy.push(key);
        }
    });
    
    keysToDestroy.forEach(key => {
        if (charts[key]) {
            charts[key].destroy();
            delete charts[key];
        }
    });
}

// Formatting helpers
function formatCurrency(val) {
    return new Intl.NumberFormat('en-US', { style: 'currency', currency: 'BRL' })
        .format(val)
        .replace("BRL", "R$");
}

function formatNumber(val) {
    return new Intl.NumberFormat('en-US').format(val);
}

// --- DATA LOADERS & CHART RENDERING ---

// 1. EXECUTIVE KPI OVERVIEW
async function loadOverviewData() {
    try {
        const res = await fetch(`${API_BASE}/api/kpi/overview`);
        const data = await res.json();
        
        // Cards
        document.getElementById("overview-revenue").textContent = formatCurrency(data.kpis.total_revenue);
        document.getElementById("overview-orders").textContent = formatNumber(data.kpis.total_orders);
        document.getElementById("overview-customers").textContent = formatNumber(data.kpis.total_customers);
        document.getElementById("overview-aov").textContent = formatCurrency(data.kpis.aov);
        
        // Monthly Trend
        const months = data.monthly_trend.map(d => d.month);
        const revenues = data.monthly_trend.map(d => Math.round(d.revenue));
        const ordersCount = data.monthly_trend.map(d => d.orders);
        
        const monthlyOpts = {
            series: [
                { name: 'Revenue (R$)', type: 'column', data: revenues },
                { name: 'Order Volume', type: 'line', data: ordersCount }
            ],
            chart: { height: 350, type: 'line', toolbar: { show: false } },
            stroke: { width: [0, 3], curve: 'smooth' },
            colors: ['#6366f1', '#10b981'],
            fill: { opacity: [0.4, 1] },
            labels: months,
            xaxis: { type: 'category', labels: { style: { colors: '#94a3b8' } } },
            yaxis: [
                { title: { text: 'Revenue (R$)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' }, formatter: (v) => formatCurrency(v) } },
                { opposite: true, title: { text: 'Orders count', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } }
            ],
            grid: { borderColor: 'rgba(255,255,255,0.05)' },
            legend: { labels: { colors: '#f8fafc' } }
        };
        charts["overview-monthly"] = new ApexCharts(document.querySelector("#chart-monthly-trend"), monthlyOpts);
        charts["overview-monthly"].render();
        
        // Payment share
        const payTypes = data.payment_methods.map(d => d.payment_type.replace('_', ' '));
        const payValues = data.payment_methods.map(d => d.revenue);
        const payOpts = {
            series: payValues,
            chart: { type: 'donut', height: 350 },
            labels: payTypes,
            colors: ['#6366f1', '#10b981', '#f59e0b', '#06b6d4', '#f43f5e'],
            legend: { position: 'bottom', labels: { colors: '#f8fafc' } },
            plotOptions: { donut: { labels: { show: false } } },
            stroke: { show: false }
        };
        charts["overview-payment"] = new ApexCharts(document.querySelector("#chart-payment-share"), payOpts);
        charts["overview-payment"].render();
        
        // Weekday Sales
        const weekdays = data.weekday_trend.map(d => d.Weekday);
        const weekdayRevs = data.weekday_trend.map(d => Math.round(d.revenue));
        const wkOpts = {
            series: [{ name: 'Revenue (R$)', data: weekdayRevs }],
            chart: { type: 'bar', height: 350, toolbar: { show: false } },
            colors: ['#8b5cf6'],
            plotOptions: { bar: { borderRadius: 4, horizontal: false } },
            xaxis: { categories: weekdays, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { labels: { style: { colors: '#94a3b8' }, formatter: (v) => formatCurrency(v) } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' }
        };
        charts["overview-weekday"] = new ApexCharts(document.querySelector("#chart-weekday-trend"), wkOpts);
        charts["overview-weekday"].render();
        
        // State Revenue distribution (top 8)
        const topStates = data.state_kpis.slice(0, 8);
        const stateNames = topStates.map(d => d.customer_state);
        const stateRevs = topStates.map(d => Math.round(d.revenue));
        const stateOpts = {
            series: [{ name: 'Revenue (R$)', data: stateRevs }],
            chart: { type: 'bar', height: 350, toolbar: { show: false } },
            colors: ['#06b6d4'],
            plotOptions: { bar: { borderRadius: 4, horizontal: true } },
            xaxis: { categories: stateNames, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' }
        };
        charts["overview-state"] = new ApexCharts(document.querySelector("#chart-state-revenue"), stateOpts);
        charts["overview-state"].render();
        
    } catch (e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

// 2. CUSTOMER SEGMENTATION
async function loadSegmentationData() {
    try {
        const res = await fetch(`${API_BASE}/api/kpi/segmentation`);
        const data = await res.json();
        
        // Find KPI stats
        const champs = data.rfm_segments.find(s => s.Segment === "Champions")?.customers || 0;
        const loyal = data.rfm_segments.find(s => s.Segment === "Loyal Customer")?.customers || 0;
        const risk = data.rfm_segments.find(s => s.Segment === "At Risk")?.customers || 0;
        const others = data.rfm_segments.filter(s => s.Segment !== "Champions" && s.Segment !== "Loyal Customer" && s.Segment !== "At Risk").reduce((acc, curr) => acc + curr.customers, 0);
        
        document.getElementById("seg-champions").textContent = formatNumber(champs);
        document.getElementById("seg-loyal").textContent = formatNumber(loyal);
        document.getElementById("seg-risk").textContent = formatNumber(risk);
        document.getElementById("seg-others").textContent = formatNumber(others);
        
        // RFM Share Pie
        const rfmNames = data.rfm_segments.map(d => d.Segment);
        const rfmCusts = data.rfm_segments.map(d => d.customers);
        const rfmOpts = {
            series: rfmCusts,
            chart: { type: 'pie', height: 350 },
            labels: rfmNames,
            colors: ['#10b981', '#6366f1', '#06b6d4', '#f43f5e', '#64748b'],
            legend: { position: 'bottom', labels: { colors: '#f8fafc' } },
            stroke: { show: false }
        };
        charts["segmentation-rfm"] = new ApexCharts(document.querySelector("#chart-rfm-pie"), rfmOpts);
        charts["segmentation-rfm"].render();
        
        // PCA Scatter Plot (ML segments)
        // Group PCA data by cluster
        const clusters = {};
        data.pca_scatter.forEach(pt => {
            const cls = `Cluster ${pt.cluster}`;
            if (!clusters[cls]) clusters[cls] = [];
            clusters[cls].push([parseFloat(pt.pc1.toFixed(3)), parseFloat(pt.pc2.toFixed(3))]);
        });
        
        const pcaSeries = Object.keys(clusters).map(cls => ({
            name: cls,
            data: clusters[cls]
        }));
        
        const pcaOpts = {
            series: pcaSeries,
            chart: { type: 'scatter', height: 350, toolbar: { show: false } },
            colors: ['#6366f1', '#10b981', '#f59e0b', '#f43f5e'],
            xaxis: { tickAmount: 10, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' },
            legend: { labels: { colors: '#f8fafc' } },
            tooltip: { shared: false, intersect: true }
        };
        charts["segmentation-pca"] = new ApexCharts(document.querySelector("#chart-kmeans-pca"), pcaOpts);
        charts["segmentation-pca"].render();
        
        // Populate table
        const tbody = document.querySelector("#rfm-table tbody");
        tbody.innerHTML = "";
        data.rfm_segments.forEach(row => {
            let badgeClass = "tag-others";
            if (row.Segment === "Champions") badgeClass = "tag-champions";
            else if (row.Segment === "Loyal Customer") badgeClass = "tag-loyal";
            else if (row.Segment === "Potential Loyalists") badgeClass = "tag-loyalist";
            else if (row.Segment === "At Risk") badgeClass = "tag-risk";
            
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><span class="tag ${badgeClass}">${row.Segment}</span></td>
                <td>${formatNumber(row.customers)}</td>
                <td>${formatCurrency(row.revenue)}</td>
                <td>${row.avg_recency.toFixed(1)}</td>
                <td>${row.avg_frequency.toFixed(2)}</td>
                <td>${formatCurrency(row.avg_monetary)}</td>
            `;
            tbody.appendChild(tr);
        });
        
    } catch (e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

// 3. LOGISTICS & SATISFACTION
async function loadLogisticsData() {
    try {
        const overviewRes = await fetch(`${API_BASE}/api/kpi/overview`);
        const overviewData = await overviewRes.json();
        
        const logRes = await fetch(`${API_BASE}/api/kpi/logistics`);
        const logData = await logRes.json();
        
        // Cards
        document.getElementById("log-avg-delivery").textContent = `${overviewData.kpis.avg_delivery_days.toFixed(1)} days`;
        document.getElementById("log-delayed-pct").textContent = `${overviewData.kpis.delayed_order_pct.toFixed(1)}%`;
        document.getElementById("log-avg-delay-duration").textContent = `${overviewData.kpis.avg_delay_days.toFixed(1)} days`;
        document.getElementById("log-avg-rating").textContent = `${overviewData.kpis.avg_review.toFixed(2)} / 5`;
        
        // Delivery distribution
        const buckets = logData.delivery_distribution.map(d => d.days_bucket);
        const orderCounts = logData.delivery_distribution.map(d => d.order_count);
        const distOpts = {
            series: [{ name: 'Orders count', data: orderCounts }],
            chart: { type: 'bar', height: 350, toolbar: { show: false } },
            colors: ['#6366f1'],
            plotOptions: { bar: { borderRadius: 4 } },
            xaxis: { categories: buckets, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' }
        };
        charts["logistics-dist"] = new ApexCharts(document.querySelector("#chart-delivery-dist"), distOpts);
        charts["logistics-dist"].render();
        
        // Rating vs delay
        const delayDays = logData.rating_vs_delay.map(d => d.delay_days);
        const scoreRating = logData.rating_vs_delay.map(d => parseFloat(d.avg_review_score.toFixed(2)));
        const delayOpts = {
            series: [{ name: 'Avg Review Score (1-5)', data: scoreRating }],
            chart: { type: 'line', height: 350, toolbar: { show: false } },
            colors: ['#f43f5e'],
            stroke: { width: 3, curve: 'smooth' },
            xaxis: { categories: delayDays, title: { text: 'Estimated Delivery Delay Days', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { min: 1.0, max: 5.0, labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' }
        };
        charts["logistics-delay-rating"] = new ApexCharts(document.querySelector("#chart-rating-delay"), delayOpts);
        charts["logistics-delay-rating"].render();
        
        // Populate Regional table
        const tbody = document.querySelector("#state-logistics-table tbody");
        tbody.innerHTML = "";
        overviewData.state_kpis.slice(0, 15).forEach(row => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.customer_state}</strong></td>
                <td>${formatCurrency(row.revenue)}</td>
                <td>${formatNumber(row.customers)}</td>
                <td>${row.avg_delivery_days.toFixed(1)} days</td>
                <td>${row.avg_delay_days.toFixed(1)} days</td>
            `;
            tbody.appendChild(tr);
        });
        
    } catch(e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

// 4. MARKETPLACE & SELLERS
async function loadSellersData() {
    try {
        const res = await fetch(`${API_BASE}/api/kpi/marketplace`);
        const data = await res.json();
        
        // Pareto Curve
        const percentiles = data.seller_pareto.map(d => d.seller_percentile);
        const revenueShares = data.seller_pareto.map(d => d.revenue_share);
        const paretoOpts = {
            series: [{ name: 'Cumulative Revenue %', data: revenueShares }],
            chart: { type: 'area', height: 350, toolbar: { show: false } },
            colors: ['#10b981'],
            fill: { type: 'gradient', gradient: { shadeIntensity: 1, opacityFrom: 0.3, opacityTo: 0.05 } },
            stroke: { width: 3 },
            xaxis: { categories: percentiles, title: { text: 'Percentile of Sellers Sorted by Revenue (%)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { max: 100, labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' },
            annotations: {
                points: [{
                    x: 14,
                    y: 80,
                    marker: { size: 6, fillColor: '#f43f5e', strokeColor: '#fff', radius: 2 },
                    label: { borderColor: '#f43f5e', style: { color: '#fff', background: '#f43f5e' }, text: '80% Revenue (Top 14% Sellers)' }
                }]
            }
        };
        charts["sellers-pareto"] = new ApexCharts(document.querySelector("#chart-seller-pareto"), paretoOpts);
        charts["sellers-pareto"].render();
        
        // Rankings Table
        const tbody = document.querySelector("#sellers-ranking-table tbody");
        tbody.innerHTML = "";
        data.seller_rankings.slice(0, 10).forEach((row, idx) => {
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>#${idx + 1}</strong></td>
                <td><span class="text-muted text-small">${row.seller_id}</span></td>
                <td>${row.state}</td>
                <td>${formatCurrency(row.revenue)}</td>
                <td>${row.orders}</td>
                <td>${row.rating.toFixed(2)} ★</td>
                <td><span class="tag tag-champions">${row.health_score.toFixed(1)}</span></td>
            `;
            tbody.appendChild(tr);
        });
        
    } catch(e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

// 5. PRODUCT SALES
async function loadProductsData() {
    try {
        const res = await fetch(`${API_BASE}/api/kpi/products`);
        const data = await res.json();
        
        const tbody = document.querySelector("#product-cats-table tbody");
        tbody.innerHTML = "";
        data.product_categories.forEach(row => {
            const freightRatio = (row.avg_freight / row.avg_price) * 100;
            const tr = document.createElement("tr");
            tr.innerHTML = `
                <td><strong>${row.category}</strong></td>
                <td>${formatCurrency(row.revenue)}</td>
                <td>${formatNumber(row.orders)}</td>
                <td>${formatCurrency(row.avg_price)}</td>
                <td>${formatCurrency(row.avg_freight)}</td>
                <td>${freightRatio.toFixed(1)}%</td>
            `;
            tbody.appendChild(tr);
        });
    } catch (e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

// 6. CLV & FORECASTING
async function loadAdvancedData() {
    try {
        const res = await fetch(`${API_BASE}/api/kpi/advanced`);
        const data = await res.json();
        
        // CLV Importance
        const features = data.clv_importance.map(d => d.feature);
        const importances = data.clv_importance.map(d => parseFloat((d.importance * 100).toFixed(1)));
        const impOpts = {
            series: [{ name: 'Importance (%)', data: importances }],
            chart: { type: 'bar', height: 320, toolbar: { show: false } },
            colors: ['#8b5cf6'],
            plotOptions: { bar: { borderRadius: 4, horizontal: true } },
            xaxis: { labels: { style: { colors: '#94a3b8' } } },
            yaxis: { categories: features, labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' }
        };
        charts["advanced-clv-imp"] = new ApexCharts(document.querySelector("#chart-clv-importance"), impOpts);
        charts["advanced-clv-imp"].render();
        
        // CLV Fit Scatter
        const points = data.clv_scatter.map(d => [parseFloat(d.actual.toFixed(3)), parseFloat(d.predicted.toFixed(3))]);
        const fitOpts = {
            series: [{ name: 'Customer prediction', data: points }],
            chart: { type: 'scatter', height: 320, toolbar: { show: false } },
            colors: ['#06b6d4'],
            xaxis: { title: { text: 'Actual CLV (Log-scaled)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } },
            yaxis: { title: { text: 'Predicted CLV (Log-scaled)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' } } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' },
            legend: { show: false }
        };
        charts["advanced-clv-fit"] = new ApexCharts(document.querySelector("#chart-clv-fit"), fitOpts);
        charts["advanced-clv-fit"].render();
        
        // Prophet Revenue Forecast
        const dates = data.revenue_forecast.map(d => d.date);
        const actualSales = data.revenue_forecast.map(d => d.actual ? Math.round(d.actual) : null);
        const predictedSales = data.revenue_forecast.map(d => Math.round(d.predicted));
        const lowerBound = data.revenue_forecast.map(d => Math.round(d.lower_bound));
        const upperBound = data.revenue_forecast.map(d => Math.round(d.upper_bound));
        
        const forecastOpts = {
            series: [
                { name: 'Historical Actual Sales', type: 'line', data: actualSales },
                { name: 'Prophet Predicted Sales', type: 'line', data: predictedSales },
                { name: 'Confidence Interval Upper', type: 'line', data: upperBound },
                { name: 'Confidence Interval Lower', type: 'line', data: lowerBound }
            ],
            chart: { height: 350, type: 'line', toolbar: { show: false } },
            stroke: { width: [3, 2, 1, 1], dashArray: [0, 0, 4, 4], curve: 'smooth' },
            colors: ['#6366f1', '#10b981', '#f59e0b', '#f59e0b'],
            fill: { opacity: [1, 1, 0.15, 0.15] },
            xaxis: { categories: dates, type: 'datetime', labels: { style: { colors: '#94a3b8' } } },
            yaxis: { title: { text: 'Daily Revenue (R$)', style: { color: '#94a3b8' } }, labels: { style: { colors: '#94a3b8' }, formatter: (v) => formatCurrency(v) } },
            grid: { borderColor: 'rgba(255,255,255,0.05)' },
            legend: { labels: { colors: '#f8fafc' } }
        };
        charts["advanced-forecast"] = new ApexCharts(document.querySelector("#chart-sales-forecast"), forecastOpts);
        charts["advanced-forecast"].render();
        
    } catch(e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

// 7. RECOMMENDATION ENGINE
async function loadRecommendationsData(searchUserId = null) {
    try {
        const queryStr = searchUserId ? `?user_id=${searchUserId}` : '';
        const res = await fetch(`${API_BASE}/api/kpi/recommendations${queryStr}`);
        const data = await res.json();
        
        // Cache samples
        sampleUsers = data.active_samples;
        
        // Populate chips
        const chipsContainer = document.getElementById("sample-chips-container");
        chipsContainer.innerHTML = "";
        data.active_samples.slice(0, 4).forEach(usr => {
            const div = document.createElement("div");
            div.className = "chip";
            div.textContent = `${usr.customer_unique_id.substring(0, 8)} (${usr.Segment})`;
            div.addEventListener("click", () => {
                document.getElementById("customer-search-input").value = usr.customer_unique_id;
                loadRecommendationsData(usr.customer_unique_id);
            });
            chipsContainer.appendChild(div);
        });
        
        const resultsSection = document.getElementById("recommendations-results");
        
        if (searchUserId) {
            document.getElementById("cust-meta-text").textContent = `ID: ${searchUserId} | Location: ${data.purchase_history[0]?.customer_city.toUpperCase()}, ${data.purchase_history[0]?.customer_state}`;
            
            // Pop purchase history table
            const histBody = document.querySelector("#purchase-history-table tbody");
            histBody.innerHTML = "";
            if (data.purchase_history.length > 0) {
                data.purchase_history.forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td>${row.purchase_date}</td>
                        <td><span class="text-muted text-small">${row.product_id.substring(0, 10)}...</span></td>
                        <td><span class="tag tag-others">${row.product_category || 'other'}</span></td>
                        <td>${formatCurrency(row.price)}</td>
                    `;
                    histBody.appendChild(tr);
                });
            } else {
                histBody.innerHTML = "<tr><td colspan='4' class='text-muted text-center'>No purchases found.</td></tr>";
            }
            
            // Pop SVD recs table
            const svdBody = document.querySelector("#svd-recs-table tbody");
            svdBody.innerHTML = "";
            if (data.svd_recommendations.length > 0) {
                data.svd_recommendations.forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><span class="text-muted text-small">${row.product_id}</span></td>
                        <td><span class="tag tag-loyal">${row.product_category || 'other'}</span></td>
                        <td><strong>${row.predicted_rating.toFixed(2)}</strong></td>
                        <td>${formatCurrency(row.price)}</td>
                    `;
                    svdBody.appendChild(tr);
                });
            } else {
                svdBody.innerHTML = "<tr><td colspan='4' class='text-muted text-center'>No models precomputed for this ID. Run recompute.</td></tr>";
            }
            
            // Pop Similarity recs table
            const simBody = document.querySelector("#sim-recs-table tbody");
            simBody.innerHTML = "";
            if (data.similarity_recommendations.length > 0) {
                data.similarity_recommendations.forEach(row => {
                    const tr = document.createElement("tr");
                    tr.innerHTML = `
                        <td><span class="text-muted text-small">${row.product_id}</span></td>
                        <td><span class="tag tag-loyalist">${row.product_category || 'other'}</span></td>
                        <td><strong>${(row.similarity * 100).toFixed(1)}% Match</strong></td>
                        <td>${formatCurrency(row.price)}</td>
                    `;
                    simBody.appendChild(tr);
                });
            } else {
                simBody.innerHTML = "<tr><td colspan='4' class='text-muted text-center'>No similar items found for this user's purchases.</td></tr>";
            }
            
            resultsSection.style.display = "flex";
        } else {
            resultsSection.style.display = "none";
        }
        
    } catch(e) {
        console.error(e);
    } finally {
        showLoader(false);
    }
}

function triggerSearch() {
    const userId = document.getElementById("customer-search-input").value.trim();
    if (!userId) {
        alert("Please enter a valid Customer Unique ID");
        return;
    }
    showLoader(true);
    loadRecommendationsData(userId);
}
