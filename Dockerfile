# Use an official Python runtime
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy all files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the port (make sure this matches the port Flask uses)
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
