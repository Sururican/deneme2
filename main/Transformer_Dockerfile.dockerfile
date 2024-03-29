# official base image of Python 
FROM python:3.10-slim

WORKDIR /app
#kopieren wir aktuelle Directory to  /App Folder
COPY . /app
RUN pip install --no-cache-dir -r requirements
# Port 5000 der HTTP Server  außerhalb dieses Container zur verfügung machen 
EXPOSE 8089

ENV NAME World 

# Transformer_Prediction.py ausführen wenn der Container gestartet wird
CMD ["python", "Transformer_Prediction.py"]



  
