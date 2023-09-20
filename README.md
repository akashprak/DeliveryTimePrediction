## Food Delivery Time Prediction using Regression

### environment created
```
conda create -p venv python==3.11

conda activate venv/
```

### Installed required libraries
```
pip install -r requirements.txt
```

### Training model
```
python src/training_pipeline.py
```

### Running flask app
```
python app.py
```

### Docker running on local
```
docker build -t deliveryprediction .

docker run -p 5000:5000 deliveryprediction
```

### Docker Setup In EC2- commands to be Executed

sudo apt-get update -y

sudo apt-get upgrade

#### required
curl -fsSL https://get.docker.com -o get-docker.sh

sudo sh get-docker.sh

sudo usermod -aG docker ubuntu

newgrp docker