# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install TensorBoard
RUN pip install tensorboard

# Make port 6006 available for TensorBoard
EXPOSE 6006

# Define environment variable
ENV PALAESTRA_CONFIG=/app/palaestra/config/settings.json

# Run app.py when the container launches
CMD ["python", "palaestra/main.py"]