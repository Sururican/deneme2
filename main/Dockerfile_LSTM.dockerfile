# official base image of Python 
FROM arm64v8/python:3.10-slim


# Install python3.10-dev package

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements

# Make port 80 available to the world outside this container
EXPOSE 80

# Define environment variable
ENV NAME World

# Run lstm.py when the container launches
CMD ["python", "LSTM_Prediction.py"]
