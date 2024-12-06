# BirdCLEF

1. Clone this repository:

```
git clone git@github.com:jsalbert/sound_classification_ml_production.git
```
2. Create a virtual environment using [virtualenv](https://virtualenv.pypa.io/en/latest/) and install library requirements for Windows:

```
pip install virtualenv
virtualenv myenv
source myenv/Scripts/activate 
pip install -r requirements.txt
```

3. Go to the folder `flask_app` and run the app locally in your computer:

```
python app.py 
```

4. Access it via [localhost:5000](http://localhost:5000/)

You should be able to see this screen, upload and classify a sound:

<p align="center">
<img src="https://github.com/ramyasri-m/BirdCLEF/blob/main/images/Frontend.png?raw=true" alt="Frontend" width="500"/>
</p>

## Deployment Steps for Dockerized Application on AWS EC2

## 1. Prepare AWS EC2 Instance:

Launch an EC2 instance on AWS with an appropriate configuration (instance type, security groups, etc.).
SSH into the instance and ensure Docker is installed 

```
RUN yum install -y docker && yum update -y docker
```

## 2. Upload Project Files to EC2:

Upload all project-related files, including source code, Dockerfiles, and configuration files to the EC2 instance using SCP or SFTP (or use GitHub integration if applicable).

## 3. Create Docker Images:
Navigate to the project directory on the EC2 instance.
Build the Docker images using the command:

```
docker build -t ec2-flask:v1.0 -f Dockerfile.txt .
```
This step compiles the application into a Docker image, which includes the Flask API, ML model, and the web server.

## 4. Run Docker Containers:

Once the Docker image is built, you can start the application container using the following command:

```
docker run -d -p 80:5000 ec2-flask:v1.0
```
This command runs the Docker container in detached mode (-d) and maps port 5000 from the container to port 80 on the EC2 instance, allowing external access to the Flask API.

## 5. Configure Security Groups:

Open port 80 (or any port your Flask app uses) in the EC2 instance security group to allow inbound HTTP traffic to the application server.

## 6. Verify Application:

Access the deployed application using the EC2 public IP address and the mapped port https://<EC2_PUBLIC_IP>:80

Test the functionality to ensure the application is running as expected.
