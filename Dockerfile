# =============================================================================
# Smart Customer Churn Prediction AI - Containerfile
# Python 3.12 Slim Linux Environment with Streamlit & Scikit-learn
# =============================================================================

# Step 1: Base Image
FROM python:3.12-slim

# Step 2: System configuration & lightweight curl utility for container health checks
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Step 3: Set Application Working Directory
WORKDIR /app

# Step 4: Python Runtime Environment Variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Step 5: Install Python Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Step 6: Copy Application Modules, Artifacts, and SQL Schemas
# (Model artifacts and metrics are baked into image for decoupled standalone inference)
COPY src/ ./src/
COPY app/ ./app/
COPY models/ ./models/
COPY sql/ ./sql/

# Step 7: Create and Switch to Non-Root User for Enhanced Container Security
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Step 8: Expose Streamlit Default Port
EXPOSE 8501

# Step 9: Define Container Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Step 10: Launch Streamlit Application Entrypoint
CMD ["python", "-m", "streamlit", "run", "app/streamlit_app.py"]
