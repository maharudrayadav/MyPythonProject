# Use an official lightweight Python image
FROM python:3.11-slim

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port (Flask default: 5000)
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
