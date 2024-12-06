FROM python:3.10.14-slim

# Set a directory for the app
WORKDIR /usr/src/flask_app

# Copy all the files to the container
COPY . .

# Install dependencies
RUN apt-get update && \
    apt-get install -y libsndfile1 && \
    apt-get clean

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Tell the port number the container should expose
EXPOSE 5000

# Run the Gunicorn command to start the app in production
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
