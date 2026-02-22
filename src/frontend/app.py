"""
Streamlit frontend for Cats vs Dogs inference.
Uploads an image and calls the FastAPI /predict endpoint.
"""

import os
from typing import Dict

import requests
import streamlit as st


DEFAULT_API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8080")
REQUEST_TIMEOUT_SEC = 30


def call_health(api_base_url: str) -> Dict:
    response = requests.get(f"{api_base_url}/health", timeout=REQUEST_TIMEOUT_SEC)
    response.raise_for_status()
    return response.json()


def call_ready(api_base_url: str) -> bool:
    response = requests.get(f"{api_base_url}/ready", timeout=REQUEST_TIMEOUT_SEC)
    return response.status_code == 200


def call_predict(
    api_base_url: str, filename: str, content: bytes, content_type: str
) -> Dict:
    files = {"file": (filename, content, content_type or "application/octet-stream")}
    response = requests.post(
        f"{api_base_url}/predict",
        files=files,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    response.raise_for_status()
    return response.json()


st.set_page_config(
    page_title="Cats vs Dogs Predictor", page_icon="🐾", layout="centered"
)
st.title("Cats vs Dogs Predictor")
st.caption("Upload a pet photo and run inference through the deployed FastAPI service.")

with st.sidebar:
    st.subheader("Service Settings")
    api_base_url = st.text_input("API Base URL", value=DEFAULT_API_BASE_URL).rstrip("/")

    if st.button("Check Service", use_container_width=True):
        try:
            health = call_health(api_base_url)
            ready = call_ready(api_base_url)
            st.success("Service is reachable.")
            st.json({"health": health, "ready": ready})
        except Exception as exc:
            st.error(f"Service check failed: {exc}")

uploaded_file = st.file_uploader(
    "Upload image", type=["jpg", "jpeg", "png", "bmp", "webp"]
)

if uploaded_file is not None:
    image_bytes = uploaded_file.getvalue()
    st.image(image_bytes, caption=uploaded_file.name, use_column_width=True)

    if st.button("Predict", type="primary", use_container_width=True):
        try:
            result = call_predict(
                api_base_url=api_base_url,
                filename=uploaded_file.name,
                content=image_bytes,
                content_type=uploaded_file.type or "image/jpeg",
            )

            st.success("Prediction completed")
            left, right = st.columns(2)
            left.metric("Label", result.get("label", "unknown"))
            right.metric("Confidence", f"{result.get('confidence', 0.0):.4f}")

            st.write("Probabilities")
            probs = result.get("probabilities", {})
            if probs:
                for label, prob in probs.items():
                    st.progress(float(prob), text=f"{label}: {float(prob):.4f}")
            st.write("Response payload")
            st.json(result)
        except requests.HTTPError as exc:
            detail = ""
            try:
                detail = exc.response.json()
            except Exception:
                detail = exc.response.text
            st.error(f"Prediction request failed: {detail}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
