# Use an official Python runtime as a base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY app.py /app/

# Expose the port the application runs on
EXPOSE 5000

# Run the application
CMD ["python", "app.py"]
