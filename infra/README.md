# Infrastructure

This is repo for all the docker files needed to develop, run and evaluate the agent.


## MLFlow 
Checkout [here](https://mlflow.org/docs/latest/self-hosting/#other-deployment-options) 

Pretty much this 
```bash
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
cp .env.dev.example .env
docker compose up -d
```
