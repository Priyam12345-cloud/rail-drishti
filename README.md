# Rail Drishti -- AI-Powered Indian Railways Intelligence Platform

> **India's smartest train travel platform** -- route planning across 5,208 trains and 8,533 stations, distance-based fare calculation for 6 classes, AI chatbot with multilingual support, and delay prediction using LightGBM + MLflow. Built entirely on Databricks.

---

## What It Does

Rail Drishti processes 417K+ Indian Railways schedule entries using PySpark to build a graph+index routing engine with city-cluster expansion (Mumbai = BCT+CSTM+LTT+BDTS). It finds direct and connecting trains, calculates distance-based fares (per-km for GEN/SL/CC/3AC/2AC/1AC), predicts delays with a LightGBM model trained on 500K+ records, and provides a multilingual AI chatbot powered by Llama 3.3 70B via Databricks Model Serving -- all deployed as a Databricks App.

---

## Architecture Diagram

```
+-----------------------------------------------------------------------+
|                          USER INTERFACE                                |
|  +------------------------+    +-----------------------------------+  |
|  |  Databricks App        |    |  Notebook UIs (Gradio)            |  |
|  |  (FastAPI, 46K LOC)    |    |  - Dashboard   - Chatbot          |  |
|  |  Routes|Delay|Chat     |    |  - Routing Engine  - Fare Calc    |  |
|  +----------+-------------+    +----------------+------------------+  |
+-----------+---------------------------------+-------------------------+
            |                                 |
+-----------v---------------------------------v-------------------------+
|                     AI / ML LAYER (Databricks)                        |
|                                                                       |
|  +---------------------------+  +----------------------------------+  |
|  |  Databricks Model Serving |  |  MLflow Experiment Tracking      |  |
|  |  (ai_query endpoint)      |  |                                  |  |
|  |                           |  |  LightGBM Delay Model            |  |
|  |  Meta Llama 3.3 70B      |  |  500K+ records, 98 trains        |  |
|  |  Instruct                 |  |  382 stations                    |  |
|  |                           |  |  mlflow.lightgbm.load_model()    |  |
|  |  - NLP Query Parsing      |  |  Model URI: runs:/d9ba760c.../   |  |
|  |  - Route Recommendation   |  |                                  |  |
|  |  - Waitlist Prediction    |  +----------------------------------+  |
|  |  - Multilingual Chat      |                                        |
|  +---------------------------+                                        |
+-------------------------------+---------------------------------------+
                                |
+-------------------------------v---------------------------------------+
|               DATA PROCESSING (Apache Spark / PySpark)                |
|                                                                       |
|  +----------------+ +----------------+ +------------------+           |
|  | df_schedule    | | df_stations    | | df_trains        |           |
|  | 417,080 stops  | | 8,990 stations | | 5,208 trains     |           |
|  | groupBy, agg   | | lat/lng coords | | class avail.     |           |
|  | filter,collect | | state, zone    | | fares, types     |           |
|  +----------------+ +----------------+ +------------------+           |
|                                                                       |
|  +-------------------------------------------------------------+     |
|  |              ROUTING ENGINE (Graph + Index)                  |     |
|  |  - 1.2M graph edges + train_stops index                     |     |
|  |  - CITY_CLUSTERS: Mumbai(6), Delhi(5), Kolkata(3)...        |     |
|  |  - Direct train search: set intersection O(1) per train     |     |
|  |  - Via-hub connecting: scored hub + time alignment           |     |
|  |  - Distance-based fares: haversine hop-by-hop x IR rates    |     |
|  +-------------------------------------------------------------+     |
|                                                                       |
|  +----------------+ +------------------------------------------+     |
|  | Bus Routes     | | PDF Knowledge Base (PyPDF2)              |     |
|  | Pan-India CSV  | | 6 railway docs (815+ pages)              |     |
|  | Multi-modal    | | SCR G&SR, Tatkal FAQ, Citizen Charter    |     |
|  +----------------+ +------------------------------------------+     |
+-----------------------------------------------------------------------+
```

---

## Databricks Technologies Used

| Technology | Usage |
|---|---|
| **Apache Spark / PySpark** | Core data processing -- 417K schedule rows with groupBy, filter, agg, countDistinct for hub detection, zone analytics, class availability |
| **Databricks Model Serving** | ai_query("databricks-meta-llama-3-3-70b-instruct") for NLP parsing, route recommendation, waitlist prediction, multilingual chatbot |
| **MLflow** | LightGBM delay prediction model -- experiment tracking + mlflow.lightgbm.load_model() for inference |
| **Databricks Apps** | FastAPI web application deployment (routes, delay prediction, AI chat, ratings) |
| **Serverless Compute** | All notebook execution on serverless interactive clusters |

## Open-Source Models

| Model | Purpose |
|---|---|
| **Meta Llama 3.3 70B Instruct** | NLP query parsing, route recommendation, multilingual chatbot (via Databricks Model Serving) |
| **LightGBM** | Train delay prediction -- trained on 500K+ historical delay records covering 98 trains, 382 stations |
| **sentence-transformers** | Text embeddings for PDF knowledge base search |

---

## How to Run

### Prerequisites
- Databricks workspace with Model Serving enabled
- Access to `databricks-meta-llama-3-3-70b-instruct` endpoint
- Python 3.10+

### Step 1: Upload Data
```
Upload the Data/ folder contents to:
/Workspace/Users/<your-email>/Data/
  - schedules.json  (417K train stops)
  - stations.json   (8,990 stations with coordinates)
  - trains.json     (5,208 trains with class info)
```

### Step 2: Upload Delay Data
```
Upload train_delay/ folder to:
/Workspace/Users/<your-email>/train_delay/
  - merged_output.csv  (500K+ delay records)
```

### Step 3: Run Notebooks (in order)
1. **Rail Drishti Routing Engine** (`notebooks/rail_drishti_routing_engine.py`)
   - Run cells 1-8: loads data, builds routing graph, runs PNBE->Mumbai demo
   - Run cells 9-11: AI layers (NLP, recommendations, delay prediction)
2. **Train Assistant Chatbot** (`notebooks/train_assistant_chatbot.py`)
   - Run all cells: loads PDFs, builds chatbot, launches Gradio UI
3. **Rail Drishti Dashboard** (`notebooks/rail_drishti_dashboard.py`)
   - Run all cells: full Gradio dashboard with routing, chat, ratings

### Step 4: Deploy the App
```bash
# In the Databricks workspace:
# 1. Navigate to UI/rail-drishti-app/
# 2. Deploy via Databricks Apps
# 3. The app serves at your workspace URL
here is the deployable link
https://rail-drishti-7474650771899073.aws.databricksapps.com/
```

---

## Demo Steps (What to Click)

### Demo 1: Train Route Search with Fares
1. Open the **Databricks App** or run **Rail Drishti Routing Engine** notebook
2. Search: **Patna to Mumbai** (or any city pair)
3. See: 10 direct trains + 179 connecting routes
4. Each result shows: GEN|SL|CC|3AC|2AC|1AC fares, distance in km, duration

### Demo 2: AI Chatbot
1. Open the **Train Assistant Chatbot** notebook
2. Ask: "I missed my Rajdhani from Delhi to Mumbai at 6pm"
3. AI parses intent -> finds routes -> recommends best option
4. Try Hindi: "पटना से मुंबई की ट्रेन बताओ"

### Demo 3: Delay Prediction
1. In the App or Chatbot, ask: "How much delay for train 12951 tomorrow?"
2. LightGBM model predicts delay per station along the route
3. Shows: avg/max/min delay, station-wise breakdown

### Demo 4: PDF Knowledge Base
1. Chatbot has 6 railway PDFs loaded (815+ pages)
2. Ask: "What is the tatkal booking time?" or "Baggage allowance rules?"
3. AI searches PDFs and cites the source document

---

## Project Structure

```
rail-drishti/
  README.md
  notebooks/
    rail_drishti_routing_engine.py   # Core routing + fare + AI layers
    train_assistant_chatbot.py       # Multilingual chatbot + PDF KB
    rail_drishti_dashboard.py        # Full Gradio dashboard
    fare_calculator.py               # Standalone fare calculation
    multi_modal_router.py            # Train + bus combined routing
    train_delay_experiment.py        # LightGBM delay model training
  app/
    app.py                           # FastAPI Databricks App (46K LOC)
    app.yaml                         # Databricks App config
    delay_stats.json                 # Pre-computed delay statistics
    requirements.txt                 # Python dependencies
  data/                              # (not in repo -- upload to workspace)
    schedules.json                   # 417K train stops (78 MB)
    stations.json                    # 8,990 stations (1.8 MB)
    trains.json                      # 5,208 trains (14 MB)
  docs/
    architecture.png                 # Architecture diagram
```

---

## Project Write-up (500 chars)

Rail Drishti is an AI-powered Indian Railways intelligence platform on Databricks. PySpark processes 417K schedule entries into a graph+index routing engine with city-cluster expansion across 8,533 stations. Features: direct+connecting train search, distance-based fare calculation (6 classes), LightGBM delay prediction (MLflow-tracked, 500K records), multilingual AI chatbot (Llama 3.3 70B via Model Serving), PDF knowledge base, and multi-modal train+bus routing. Deployed as a Databricks App.

---

## Team

Built for the Databricks Hackathon 2026.

---

## Bonus: Quantitative Metrics & MLflow Logs

### 1. MLflow Experiment Logs

**3 Experiments tracked, 7 runs total:**

#### Experiment: Train Delay Prediction (LightGBM)
| Run | Test MAE | Test RMSE | Test R² | Status |
|-----|----------|-----------|---------|--------|
| Best (tuned) | **19.06 min** | 38.86 min | **0.783** | FINISHED |
| Hypertuned v2 | 18.75 min | 32.36 min | 0.779 | FINISHED |
| Baseline | 26.21 min | 50.86 min | 0.453 | FINISHED |

**Best Model Hyperparameters (MLflow-tracked):**
- Algorithm: LightGBM (boosting_type=gbdt)
- learning_rate: 0.230, n_estimators: 700, max_depth: 9
- num_leaves: 149, min_child_samples: 18
- Regularization: reg_alpha=0.060, reg_lambda=0.061
- Feature sampling: colsample_bytree=0.955, subsample=0.687
- Model URI: `runs:/d9ba760cbb1f42f697214e19ba2095c9/model`

#### Experiment: Mumbai Multi-Modal Route Optimizer
| Run | Origin | Destination | Routes Found | Best Time | Stops |
|-----|--------|-------------|--------------|-----------|-------|
| 1 | Mumbai CST | Thane (East) | 1 | 45 min | 17 |
| 2 | Mumbai CST | Thane (East) | 3 | 63 min | 27 |

---

### 2. Routing Engine Accuracy Metrics

| Metric | Value |
|--------|-------|
| Schedule entries processed | 417,080 |
| Stations indexed | 8,533 |
| Graph edges built | 1,235,521 |
| Hub stations (10+ trains) | 6,470 |
| Trains with stop-lists | 5,208 |
| City clusters defined | 15 cities, 45+ codes |
| Direct train recall (PNBE→Mumbai) | 10/10 (was 0 without clusters) |
| Via-hub routes (PNBE→Mumbai) | 179 connecting options |
| Fare calculation coverage | 6 classes (GEN/SL/CC/3AC/2AC/1AC) |
| Distance method | Hop-by-hop haversine × 1.2 rail factor |

---

### 3. Delay Prediction Model Performance

**Dataset:** 500K+ records, 98 trains, 382 stations

| Metric | Baseline | Tuned Model | Improvement |
|--------|----------|-------------|-------------|
| MAE | 26.21 min | 19.06 min | 27% better |
| RMSE | 50.86 min | 38.86 min | 24% better |
| R² Score | 0.453 | 0.783 | +73% |

**Features used:** train_no, station (encoded), day_of_week, month, is_holiday

---

### 4. Multilingual Chatbot Evaluation

The chatbot uses `databricks-meta-llama-3-3-70b-instruct` via Databricks Model Serving.

**Languages tested:**

| Language | Sample Query | Response Quality | Correct Routing |
|----------|-------------|------------------|-----------------|
| English | "Delhi to Mumbai trains" | Fluent, detailed | Yes |
| Hindi | "ट्रेन में बाथरूम कहाँ है?" | Fluent Hindi reply | N/A (FAQ) |
| Tamil | "விபத்து எப்படி நடக்கும்?" | Tamil response | N/A (FAQ) |
| Spanish | "¿Dónde está el baño?" | Spanish response | N/A (FAQ) |
| Bengali | "ট্রেন কখন আসবে?" | Bengali response | Yes |
| Marathi | "मुंबई ते पुणे ट्रेन" | Marathi response | Yes |

**PDF Knowledge Base:** 6 documents, 815+ pages (SCR G&SR, Tatkal FAQ, Citizen Charter, e-Ticket FAQ, Internet Tickets guide, General Rules 2015)

---

### 5. AI Model Performance

| AI Function | Model | Latency | Accuracy |
|-------------|-------|---------|----------|
| NLP Query Parsing | Llama 3.3 70B | ~2s | Correctly extracts origin/dest/time from natural language |
| Route Recommendation | Llama 3.3 70B | ~3s | Considers delay risk, connection safety, cost trade-offs |
| Waitlist Prediction | Llama 3.3 70B | ~2s | Probabilistic assessment based on train type, class, days |
| Delay Prediction | LightGBM (MLflow) | <100ms | R²=0.783, MAE=19 min across 98 trains |
| PDF Q&A | Llama 3.3 70B | ~2s | Searches 815+ pages, cites source document |
