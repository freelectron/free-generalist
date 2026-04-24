# Infrastructure

This is repo for all extra services needed to develop, run and evaluate the agent.


## Playwright

Agent's parsing capabilities require playwright if we are using `crawl4ai` package. 

## MLFlow 
Checkout [here](https://mlflow.org/docs/latest/self-hosting/#other-deployment-options) 

Pretty much this 
```bash
git clone https://github.com/mlflow/mlflow.git
cd docker-compose
cp .env.dev.example .env
docker compose up -d
```


## MongoDB 

Implemented to store text info as key-value. 
