from __future__ import annotations  # optional now, but fine to leave

import os
import mlflow
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from app import featurize_single, predict_logprice

app = FastAPI(title="Sleeper Truck Price API")

class TruckFeatures(BaseModel):
    year: Optional[int] = None
    mileage: Optional[float] = None
    horsepower: Optional[float] = None
    bunk_count: Optional[int] = None
    manufacturer: Optional[str] = None
    model: Optional[str] = None
    sleeper_type: Optional[str] = None
    transmission_type: Optional[str] = None
    transmission_make: Optional[str] = None
    engine_make: Optional[str] = None
    engine_model: Optional[str] = None
    state: Optional[str] = None
    has_apu: bool = False

@app.post("/predict")
def predict_price(input: TruckFeatures):
    X = featurize_single(
        input.year,
        input.mileage,
        input.horsepower,
        input.bunk_count,
        input.manufacturer,
        input.model,
        input.sleeper_type,
        "",  # free-text transmission
        input.transmission_type,
        input.transmission_make,
        input.engine_make,
        input.engine_model,
        input.state,
        input.has_apu,
    )

    log_pred = predict_logprice(X)
    price = float(np.exp(log_pred)[0])

    return {
        "predicted_price": price,
        "price_low_estimate": price * 0.933,
        "price_high_estimate": price * 1.067,
    }
