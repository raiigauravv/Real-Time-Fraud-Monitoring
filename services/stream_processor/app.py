import os, json, csv, time, logging, collections
from datetime import datetime
import numpy as np, joblib, requests
from kafka import KafkaConsumer
from kafka.errors import NoBrokersAvailable
import shap
import sqlalchemy as sa

from featurizer import to_features, FEATURE_NAMES
from threshold import load_threshold

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

BROKER       = os.getenv("KAFKA_BROKER","kafka:9092")
TOPIC        = os.getenv("TOPIC_TRANSACTIONS","transactions")
ALERTS_SINK  = os.getenv("ALERTS_SINK","/app/data/alerts.csv")
LATEST_SINK  = os.getenv("LATEST_SINK","/app/data/latest.csv")
DRIFT_SINK   = os.getenv("DRIFT_SINK","/app/data/drift.json")
SLACK        = os.getenv("SLACK_WEBHOOK_URL","").strip()
MODEL_PATH   = os.getenv("MODEL_PATH","/app/model.pkl")
META_PATH    = MODEL_PATH.replace(".pkl","_meta.json")
DB_URL       = os.getenv("DB_URL","")

# ── Load model metadata (F1, threshold used at training time) ─────────────────
MODEL_META = {}
if os.path.exists(META_PATH):
    with open(META_PATH) as _f:
        MODEL_META = json.load(_f)
    logger.info("Model meta: %s", MODEL_META)

# ── Drift tracking — rolling window of last 500 transactions ─────────────────
_WINDOW = 500
_score_window: collections.deque = collections.deque(maxlen=_WINDOW)
_drift_counter = 0
_DRIFT_WRITE_EVERY = 50   # write drift.json every N transactions

def ensure_csv(path, headers):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        with open(path,"w",newline="", encoding='utf-8') as f: 
            csv.writer(f).writerow(headers)

ensure_csv(ALERTS_SINK, ["ts","prob","Amount",*FEATURE_NAMES,"top_features_json"])
ensure_csv(LATEST_SINK, ["ts","prob","Amount",*FEATURE_NAMES,"is_alert"])

model = joblib.load(MODEL_PATH)

# For VotingClassifier, use the first estimator (XGBoost) for SHAP
try:
    _shap_model = model.estimators_[0] if hasattr(model, "estimators_") else model
    explainer = shap.TreeExplainer(_shap_model)
    SHAP_OK = True
except Exception as e:
    logger.warning("SHAP unavailable: %s", e)
    SHAP_OK = False

engine=None
if DB_URL:
    try:
        engine=sa.create_engine(DB_URL, pool_pre_ping=True)
        with engine.connect() as c: c.execute(sa.text("SELECT 1"))
        logger.info("DB connected")
    except Exception as e:
        logger.warning("DB connect failed: %s", e); engine=None

def create_consumer_with_retry():
    max_retries = 30
    retry_interval = 5
    
    for attempt in range(max_retries):
        try:
            logger.info("Attempting to connect to Kafka at %s (attempt %d)", BROKER, attempt + 1)
            consumer = KafkaConsumer(
                TOPIC, bootstrap_servers=[BROKER],
                value_deserializer=lambda m: json.loads(m.decode()),
                auto_offset_reset="latest", enable_auto_commit=True,
                group_id="fraud_processor",
                request_timeout_ms=30000,
                retry_backoff_ms=1000
            )
            logger.info("Successfully connected to Kafka")
            return consumer
        except NoBrokersAvailable:
            logger.warning("Kafka not available yet. Retrying in %d seconds...", retry_interval)
            time.sleep(retry_interval)
        except Exception as e:
            logger.error("Error connecting to Kafka: %s. Retrying in %d seconds...", e, retry_interval)
            time.sleep(retry_interval)
    
    raise RuntimeError(f"Failed to connect to Kafka after {max_retries} attempts")

consumer = create_consumer_with_retry()

def top_shap(x):
    if not SHAP_OK: return []
    vals = explainer.shap_values(x.reshape(1,-1))
    sv = vals if isinstance(vals,np.ndarray) else vals[1]
    sv = sv[0]
    idx = np.argsort(np.abs(sv))[::-1][:5]
    return [{"feature":FEATURE_NAMES[i],"contribution":float(sv[i])} for i in idx]

def write_drift(prob: float):
    """
    Track score drift: compare rolling mean of fraud probability against
    the training baseline stored in model_meta.json.
    PSI-lite: flag if rolling mean deviates >2× from baseline fraud rate.
    """
    global _drift_counter
    _score_window.append(prob)
    _drift_counter += 1
    if _drift_counter % _DRIFT_WRITE_EVERY != 0:
        return
    if len(_score_window) < 10:
        return

    scores = list(_score_window)
    rolling_alert_rate = float(np.mean([s >= load_threshold() for s in scores]))
    baseline_fraud_rate = MODEL_META.get("fraud_rate_pct", 0.172) / 100.0
    drift_ratio = rolling_alert_rate / (baseline_fraud_rate + 1e-9)

    drift_status = "stable"
    if drift_ratio > 3.0:
        drift_status = "high"
    elif drift_ratio > 1.5:
        drift_status = "elevated"

    drift_payload = {
        "ts": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "window_size": len(scores),
        "rolling_alert_rate": round(rolling_alert_rate, 4),
        "baseline_fraud_rate": round(baseline_fraud_rate, 4),
        "drift_ratio": round(drift_ratio, 3),
        "drift_status": drift_status,
        "mean_score": round(float(np.mean(scores)), 4),
        "p95_score":  round(float(np.percentile(scores, 95)), 4),
        "model_f1":   MODEL_META.get("f1", "N/A"),
        "model_name": MODEL_META.get("model", "unknown"),
    }
    os.makedirs(os.path.dirname(DRIFT_SINK), exist_ok=True)
    with open(DRIFT_SINK, "w") as fh:
        json.dump(drift_payload, fh, indent=2)

    if drift_status != "stable":
        logger.warning("DRIFT %s: ratio=%.2f rolling_rate=%.4f",
                       drift_status, drift_ratio, rolling_alert_rate)


def slack(prob, amount, ts, top):
    if not SLACK: return
    try:
        drivers = ", ".join(f"{d['feature']}={d['contribution']:.3f}" for d in top)
        requests.post(SLACK, data=json.dumps({"text": f":rotating_light: FRAUD {prob:.2f} | amt={amount:.2f} | ts={ts} | {drivers}"}))
    except Exception as e:
        logger.warning("slack failed: %s", e)

logger.info("processor running...")
for msg in consumer:
    rec = msg.value
    ts = datetime.utcnow().isoformat(timespec="seconds")+"Z"
    x = to_features(rec)
    if hasattr(model,"predict_proba"): prob=float(model.predict_proba(x.reshape(1,-1))[0,1])
    else: prob=float(model.predict(x.reshape(1,-1))[0])
    amount = float(rec.get("Amount",0.0))
    th = load_threshold()
    is_alert = prob >= th

    # Track drift on every transaction
    write_drift(prob)

    with open(LATEST_SINK,"a",newline="", encoding='utf-8') as f:
        csv.writer(f).writerow([ts,prob,amount,*[rec.get(n,"") for n in FEATURE_NAMES], int(is_alert)])

    if is_alert:
        top = top_shap(x)
        with open(ALERTS_SINK,"a",newline="", encoding='utf-8') as f:
            csv.writer(f).writerow([ts,prob,amount,*[rec.get(n,"") for n in FEATURE_NAMES], json.dumps(top)])
        slack(prob, amount, ts, top)
        if engine:
            try:
                with engine.begin() as conn:
                    conn.execute(sa.text("""
                        INSERT INTO alerts (ts, prob, amount, features, shap)
                        VALUES (:ts, :prob, :amount, :features, :shap)
                    """), {"ts":ts, "prob":prob, "amount":amount,
                           "features": json.dumps({k: rec.get(k) for k in FEATURE_NAMES}),
                           "shap": json.dumps(top)})
            except Exception as e:
                logger.warning("DB insert failed: %s", e)
