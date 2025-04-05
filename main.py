from fastapi import FastAPI, HTTPException
import torch
import yaml
import mlflow.pytorch
from model import PINN
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

def load_config():
    try:
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully for API.")
        return config
    except FileNotFoundError:
        logger.error("config.yaml file not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config.yaml: {e}")
        raise

def load_run_id():
    try:
        with open("run_id.txt", "r") as f:
            run_id = f.read().strip()
        logger.info(f"Loaded Run ID: {run_id}")
        return run_id
    except FileNotFoundError:
        logger.error("run_id.txt file not found. Please run train.py first.")
        raise
    except Exception as e:
        logger.error(f"Error loading run_id: {e}")
        raise

def load_model(config, run_id):
    try:
        DEVICE = torch.device(config["training"]["device"] if torch.cuda.is_available() else "cpu")
        model_uri = f"runs:/{run_id}/pinn_model"
        model = mlflow.pytorch.load_model(model_uri).to(DEVICE)
        logger.info("Model loaded successfully from MLflow.")
        return model, DEVICE
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {e}")
        raise

# Load config, run_id, and model at startup
try:
    config = load_config()
    run_id = load_run_id()
    model, DEVICE = load_model(config, run_id)
except Exception as e:
    logger.critical(f"Failed to initialize API: {e}")
    raise

@app.get("/predict")
async def predict(x: float):
    try:
        x_tensor = torch.tensor([x], dtype=torch.float32).view(-1, 1).to(DEVICE)
        prediction = model.predict(x_tensor)
        logger.info(f"Prediction made for x={x}: {prediction[0, 0]}")
        return {"prediction": float(prediction[0, 0])}
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    try:
        logger.debug("Health check requested.")
        return {"status": "healthy"}
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")