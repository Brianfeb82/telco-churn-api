from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import pickle
import numpy as np
import pandas as pd
import secrets
import os
import logging
import sqlite3
from datetime import datetime
from dotenv import load_dotenv

# 0. Load Configuration
load_dotenv()
API_USER = os.getenv("API_USERNAME", "admin")
API_PASS = os.getenv("API_PASSWORD", "admin123")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./predictions.db")

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("churn_api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("churn_api")

# Helper to get DB connection
def get_db_connection():
    if DATABASE_URL.startswith("sqlite"):
        # Strip sqlite:/// prefix
        db_path = DATABASE_URL.replace("sqlite:///", "")
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row # For dictionary-like access in SQLite
        return conn
    else:
        # Use psycopg2 for Postgres
        try:
            import psycopg2
            conn = psycopg2.connect(DATABASE_URL)
            # For dictionary-like access in psycopg2
            from psycopg2.extras import RealDictCursor
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            conn.cursor = lambda: cursor # Monkey patch to return RealDictCursor
            return conn
        except ImportError:
            logger.error("psycopg2 not installed. Cannot connect to PostgreSQL.")
            raise Exception("psycopg2 not installed for PostgreSQL connection.")
        except Exception as e:
            logger.error(f"Error connecting to PostgreSQL: {e}")
            raise

# 1. Load Model and Scaler using Lifespan (FastAPI 0.93+)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Determine basic directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, 'model_churn_v2.pkl')
    SCALER_PATH = os.path.join(BASE_DIR, 'scaler_v2.pkl')
    
    logger.info(f"🚀 Starting up... DB Mode: {'SQLite' if DATABASE_URL.startswith('sqlite') else 'PostgreSQL'}")
    
    # Init Database
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # SQL syntax differences: Postgres uses SERIAL, SQLite uses AUTOINCREMENT
        pk_type = "SERIAL PRIMARY KEY" if not DATABASE_URL.startswith("sqlite") else "INTEGER PRIMARY KEY AUTOINCREMENT"
        timestamp_type = "TIMESTAMP" if not DATABASE_URL.startswith("sqlite") else "DATETIME"
        
        cursor.execute(f'''
            CREATE TABLE IF NOT EXISTS churn_history (
                id {pk_type},
                tenure INTEGER,
                monthly_charges REAL,
                total_charges REAL,
                prediction TEXT,
                probability REAL,
                "user" TEXT,
                timestamp {timestamp_type}
            )
        ''')
        conn.commit()
        conn.close()
        logger.info("✅ Database check complete.")
    except Exception as e:
        logger.error(f"❌ Database initialization error: {e}")

    # Load Assets
    try:
        # 2. Load Model
        if os.path.exists(MODEL_PATH):
            with open(MODEL_PATH, 'rb') as f:
                app.state.model = pickle.load(f)
            logger.info("Model loaded successfully.")
        else:
            logger.error(f"Error: Model file not found at {MODEL_PATH}")
            app.state.model = None

        # Load Scaler
        if os.path.exists(SCALER_PATH):
            with open(SCALER_PATH, 'rb') as f:
                app.state.scaler = pickle.load(f)
            print("Scaler loaded successfully.")
        else:
            print(f"Error: Scaler file not found at {SCALER_PATH}")
            app.state.scaler = None
            
    except Exception as e:
        print(f"CRITICAL ERROR during model loading: {e}")
        app.state.model = None
        app.state.scaler = None
    
    yield
    # Shutdown logic (optional)
    print("Shutting down API server...")

# 2. Initialize FastAPI app
app = FastAPI(
    title="Telco Churn Prediction API",
    description="API for predicting customer churn using an optimized XGBoost model.",
    version="1.0.0",
    lifespan=lifespan
)

# 3. Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 4. Security Setup
security = HTTPBasic()

def authenticate(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, API_USER)
    correct_password = secrets.compare_digest(credentials.password, API_PASS)
    if not (correct_username and correct_password):
        raise HTTPException(
            status_code=401,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

# 5. Define input data model
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float

    model_config = {
        "json_schema_extra": {
            "example": {
                "tenure": 12,
                "monthly_charges": 70.0,
                "total_charges": 840.0
            }
        }
    }

# 6. Endpoints
@app.get("/")
def home():
    return {"message": "Welcome to Telco Churn Prediction API. (Cloud DB Ready)"}

@app.post("/predict")
def predict_churn(data: CustomerData, request: Request, username: str = Depends(authenticate)):
    logger.info(f"Request by {username} | Data: {data.model_dump()}")
    
    # Retrieve model and scaler from app state
    model = getattr(request.app.state, 'model', None)
    scaler = getattr(request.app.state, 'scaler', None)
    
    if model is None or scaler is None:
        logger.error("Model or Scaler not found in App State!")
        raise HTTPException(
            status_code=500, 
            detail="Model or Scaler not loaded on server. Check terminal logs."
        )
    
    try:
        # Prepare input for prediction
        input_array = np.array([[data.tenure, data.monthly_charges, data.total_charges]])
        
        # Scaling
        input_scaled = scaler.transform(input_array)
        
        # Prediction
        prediction = int(model.predict(input_scaled)[0])
        probability = float(model.predict_proba(input_scaled)[0][1])
        status = "Churn" if prediction == 1 else "Loyal"
        
        logger.info(f"Result for {username}: {status} (Prob: {probability:.4f})")

        # 7. Save to Database (Flexible)
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Syntax difference: ? for SQLite, %s for Postgres
            mark = "?" if DATABASE_URL.startswith("sqlite") else "%s"
            
            cursor.execute(f'''
                INSERT INTO churn_history 
                (tenure, monthly_charges, total_charges, prediction, probability, "user", timestamp)
                VALUES ({mark}, {mark}, {mark}, {mark}, {mark}, {mark}, {mark})
            ''', (data.tenure, data.monthly_charges, data.total_charges, 
                  status, round(probability, 4), username, datetime.now()))
            conn.commit()
            conn.close()
            logger.info(f"Transaction saved to database mode: {'SQLite' if DATABASE_URL.startswith('sqlite') else 'PostgreSQL'}")
        except Exception as db_err:
            logger.error(f"Database save error: {db_err}")

        return {
            "prediction": status,
            "churn_probability": round(probability, 4),
            "status_code": 200,
            "user": username
        }
    except Exception as e:
        logger.error(f"Prediction error for {username}: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/history")
def get_history(limit: int = 10, username: str = Depends(authenticate)):
    """Menampilkan riwayat prediksi terbaru dari database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Placeholder difference
        mark = "?" if DATABASE_URL.startswith("sqlite") else "%s"
        
        cursor.execute(f'SELECT * FROM churn_history ORDER BY id DESC LIMIT {mark}', (limit,))
        rows = cursor.fetchall()
        
        # Format results consistently
        history = [dict(row) for row in rows]
        conn.close()
        return {"history": history, "count": len(history), "mode": "Cloud" if not DATABASE_URL.startswith("sqlite") else "Local"}
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        raise HTTPException(status_code=500, detail="Could not retrieve history.")

# 8. Run the API
if __name__ == "__main__":
    import uvicorn
    # Using reload=True is great for development
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
