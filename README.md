# trinova-ml-service

AI microservice for the Trinova ERP platform. Built with **FastAPI** and **XGBoost**, it predicts supplier delay risk and plugs into the existing .NET ERP backend as an isolated ML layer.

---

## Architecture

```
trinova-erp-frontend (Next.js)
        ↓
trinova-erp-backend (.NET API)
        ↓  HTTP
trinova-ml-service  (FastAPI + XGBoost)   ← this repo
```

---

## Project structure

```
trinova-ml-service/
├── app/
│   ├── main.py                   # FastAPI app factory + lifespan
│   ├── routes/
│   │   └── prediction.py         # POST /predict/supplier-risk, POST /train
│   ├── services/
│   │   ├── preprocess.py         # Feature engineering pipeline
│   │   ├── train_model.py        # XGBoost training + joblib persistence
│   │   └── predict.py            # Inference + model cache
│   ├── models/
│   │   └── xgboost_model.pkl     # Generated at runtime (auto-trained on cold start)
│   └── datasets/
│       └── supplier_training.csv # Bundled training dataset
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment variables
copy .env.example .env

# 4. Run the service
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The model is **auto-trained on first startup** using the bundled dataset.  
Interactive docs: http://localhost:8000/docs

---

## Endpoints

### `GET /health`
Returns service status and whether a trained model is loaded.

```json
{
  "status": "ok",
  "model_ready": true,
  "model_path": "...app/models/xgboost_model.pkl"
}
```

---

### `POST /predict/supplier-risk`
Predict the delay risk for a supplier.

**Request body:**
```json
{
  "supplier_id": 12,
  "supplier_price": 500000,
  "lead_time_days": 3,
  "claim_rate": 0.02,
  "on_time_rate": 0.95,
  "order_frequency": 24
}
```

**Response:**
```json
{
  "supplier_id": 12,
  "risk_level": "LOW",
  "delay_probability": 0.12,
  "late_probability": 12
}
```

Risk levels: `LOW` (< 30%), `MEDIUM` (30–59%), `HIGH` (≥ 60%)

---

### `POST /train`
Retrain using the bundled dataset, or a CSV path on the server.

**Request body (optional):**
```json
{ "csv_path": null }
```

---

### `POST /train/from-rows`
Send live rows from the ERP database as JSON. The .NET backend queries SQL, aggregates per supplier, and posts here.

**Request body:**
```json
{
  "append_to_existing": true,
  "rows": [
    {
      "supplier_price": 320000,
      "lead_time_days": 5,
      "claim_rate": 0.05,
      "on_time_rate": 0.88,
      "order_frequency": 12,
      "late_delivery": 1
    }
  ]
}
```

Set `append_to_existing: true` (default) to merge with the bundled historical dataset — this prevents the model from forgetting older patterns when there are few new rows.

---

### `POST /train/from-csv-upload`
Upload a CSV file directly (multipart/form-data). Useful for manual retraining from an ERP export.

```
POST /train/from-csv-upload?append_to_existing=true
Content-Type: multipart/form-data
file: <your_export.csv>
```

---

**All three `/train` variants return:**
```json
{
  "message": "Model trained and saved successfully.",
  "accuracy": 0.9167,
  "roc_auc": 0.9722,
  "samples_trained": 96,
  "samples_tested": 24,
  "model_path": "...xgboost_model.pkl",
  "classification_report": "...",
  "data_source": "live_data (85 new rows) + bundled (120 rows)"
}
```

---

## Training dataset columns

| Feature            | Source          | Description                       |
|--------------------|-----------------|-----------------------------------|
| `supplier_price`   | Catalog         | Unit / contract price             |
| `lead_time_days`   | Catalog         | Agreed lead time                  |
| `claim_rate`       | Purchase Return | Defect / claim frequency          |
| `on_time_rate`     | GR              | Historical on-time delivery rate  |
| `order_frequency`  | PO              | Orders per year                   |
| `late_delivery`    | GR (label)      | 1 = late, 0 = on time             |

---

## Calling from .NET backend

### Prediction
```csharp
var payload = new
{
    supplier_id = supplierId,
    supplier_price = supplierPrice,
    lead_time_days = leadTimeDays,
    claim_rate = claimRate,
    on_time_rate = onTimeRate,
    order_frequency = orderFrequency,
};

var response = await httpClient.PostAsJsonAsync(
    "http://localhost:8000/predict/supplier-risk",
    payload
);

var result = await response.Content.ReadFromJsonAsync<SupplierRiskResult>();
```

### Retraining with live ERP data

**Step 1 — SQL query to build training rows:**
```sql
SELECT
    s.supplier_id,
    AVG(c.unit_price)                                          AS supplier_price,
    AVG(c.lead_time_days)                                      AS lead_time_days,
    CAST(COUNT(pr.id) AS FLOAT) / NULLIF(COUNT(po.id), 0)     AS claim_rate,
    CAST(SUM(CASE WHEN gr.actual_date <= gr.expected_date
                  THEN 1 ELSE 0 END) AS FLOAT)
        / NULLIF(COUNT(gr.id), 0)                              AS on_time_rate,
    COUNT(po.id)                                               AS order_frequency,
    MAX(CASE WHEN gr.actual_date > gr.expected_date
             THEN 1 ELSE 0 END)                                AS late_delivery
FROM suppliers s
JOIN catalog           c  ON c.supplier_id  = s.supplier_id
JOIN purchase_orders  po  ON po.supplier_id = s.supplier_id
LEFT JOIN goods_receipts   gr ON gr.po_id = po.id
LEFT JOIN purchase_returns pr ON pr.po_id = po.id
WHERE po.created_at >= DATEADD(YEAR, -2, GETDATE())
GROUP BY s.supplier_id
HAVING COUNT(po.id) >= 3
```

**Step 2 — POST rows to the ML service:**
```csharp
var rows = trainingData.Select(r => new
{
    supplier_price  = r.SupplierPrice,
    lead_time_days  = r.LeadTimeDays,
    claim_rate      = r.ClaimRate,
    on_time_rate    = r.OnTimeRate,
    order_frequency = r.OrderFrequency,
    late_delivery   = r.LateDelivery,   // 0 or 1
}).ToList();

var retrain = new { rows, append_to_existing = true };

var trainResponse = await httpClient.PostAsJsonAsync(
    "http://localhost:8000/train/from-rows",
    retrain
);

var metrics = await trainResponse.Content.ReadFromJsonAsync<TrainResponse>();
// metrics.Accuracy, metrics.RocAuc, metrics.DataSource ...
```

---

## Notes

- **AHP-TOPSIS stays in the .NET backend.** This service is additive intelligence only.
- The model artifact (`xgboost_model.pkl`) is excluded from version control via `.gitignore`.
- Call `POST /train` after importing fresh ERP data to keep predictions accurate.
