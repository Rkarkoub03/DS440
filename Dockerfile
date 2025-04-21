
# Use official Python slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install system dependencies
RUN apt-get update && apt-get install -y build-essential

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Environment variable for Google credentials (can be overridden in Cloud Run)
ENV GOOGLE_APPLICATION_CREDENTIALS=/app/gcs-key.json

# Expose the port used by Flask
EXPOSE 8080

# Run the app
CMD ["python", "app/app.py"]
