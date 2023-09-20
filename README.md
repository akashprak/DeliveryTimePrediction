## Food Delivery Time Prediction using Regression
This is a Webapp using python which predicts the time it takes to deliver food from the time of order.
Machine Learning techniques like Linear Regression is used to predict the target.
The model is trained on a dataset with over 45000 samples.

It is hosted on AWS using EC2 and ECR, and included steps for the same.

### steps:
### environment created
```
conda create -p venv python==3.11

conda activate venv/
```

### Installing required libraries
```
pip install -r requirements.txt
```

### Training model
```
python src/training_pipeline.py
```

### Running the flask app
```
python app.py
```

### Docker running on local
```
docker build -t deliveryprediction .

docker run -p 7000:7000 deliveryprediction
```
### AWS
A new repostory is created and it's configurations are added as repository secret.
EC2 instance is launched in ubuntu

### Docker Setup In EC2- commands to be Executed
sudo apt-get update -y

sudo apt-get upgrade

#### required
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker

### Github runner
A self hosted runner is configured on the instance
( the commands to configure the runner is found on github - settings > actions > runners)