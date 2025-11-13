# Use a lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and the models folder
COPY app/ app/
COPY models/ models/

# Expose FastAPI port
EXPOSE 8080

# (Optional) Environment variable for clarity
ENV MODELS_DIR=/app/models

# Run the FastAPI app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
