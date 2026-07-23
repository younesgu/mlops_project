# Azure MLOps Project — Iris Classifier (End-to-End)

A minimal but complete MLOps project on **Azure Machine Learning**, using:
- A simple, reproducible ML use case (Iris classification with scikit-learn)
- **Azure ML CLI v2** (YAML-based jobs & pipelines — no heavy SDK code)
- **GitHub Actions** for CI (lint + unit tests) and CD (train, register, deploy on Azure)

It's intentionally small so you can read every file in a few minutes, but it covers every stage
a real MLOps pipeline needs.

---

## 1. Project structure

```
azure-mlops-project/
├── .github/workflows/
│   ├── ci.yml              # Lint + unit tests on every push/PR
│   └── cd.yml              # Submit Azure ML pipeline, register + deploy model (on push to main)
├── config/
│   └── workspace_config.json   # Azure ML workspace reference (placeholders)
├── data/raw/                   # Local sample data (or pulled from Azure ML data asset)
├── environment/
│   └── conda.yml            # Training/serving environment definition
├── pipelines/
│   └── training_pipeline.yml   # Azure ML CLI v2 pipeline: prep -> train -> evaluate
├── deployment/
│   ├── endpoint.yml          # Managed online endpoint definition
│   └── deployment.yml        # Deployment (model + environment + instance) definition
├── src/
│   ├── data_prep.py          # Load & split data
│   ├── train.py              # Train the model
│   ├── evaluate.py           # Evaluate + gate on accuracy threshold
│   ├── register_model.py     # Register model in Azure ML registry
│   └── score.py              # Inference entry script for the online endpoint
├── tests/
│   └── test_data_prep.py     # Example unit test
├── requirements.txt
└── README.md
```

## 2. The ML problem

Classic Iris flower classification (`sklearn.datasets.load_iris`), trained with a
`RandomForestClassifier`. Simple on purpose — the point of this repo is the **pipeline**,
not the model.

## 3. How the pieces fit together

```
        push / PR                         push to main (after CI passes)
            │                                        │
            ▼                                        ▼
   ┌─────────────────┐                     ┌────────────────────────┐
   │   CI workflow    │                     │      CD workflow        │
   │  - flake8 lint   │                     │  - az login (OIDC)      │
   │  - pytest        │                     │  - az ml job create     │
   └─────────────────┘                     │    (training_pipeline)  │
                                            │  - wait for completion   │
                                            │  - register model       │
                                            │  - create/update         │
                                            │    online endpoint       │
                                            │  - deploy new model rev  │
                                            └────────────────────────┘
```

- **CI** runs on every push/PR to any branch — fast feedback, no Azure needed.
- **CD** runs on push to `main` — talks to real Azure ML, submits the pipeline job
  defined in `pipelines/training_pipeline.yml`, and if the model passes the accuracy
  gate in `evaluate.py`, it gets registered and deployed to a managed online endpoint.

## 4. One-time Azure setup

```bash
# 1. Login & set subscription
az login
az account set --subscription "<SUBSCRIPTION_ID>"

# 2. Create resource group + Azure ML workspace (skip if you already have one)
az group create -n rg-mlops-demo -l eastus
az ml workspace create -n mlw-mlops-demo -g rg-mlops-demo

# 3. Create a compute cluster for training
az ml compute create -n cpu-cluster --type AmlCompute --min-instances 0 \
  --max-instances 2 --size Standard_DS3_v2 -g rg-mlops-demo -w mlw-mlops-demo

# 4. Create a Service Principal for GitHub Actions (OIDC federated credential recommended)
az ad sp create-for-rbac --name "sp-mlops-github" --role Contributor \
  --scopes /subscriptions/<SUBSCRIPTION_ID>/resourceGroups/rg-mlops-demo \
  --sdk-auth
```

Then in your GitHub repo, add these **secrets** (Settings → Secrets and variables → Actions):

| Secret name              | Value                                   |
|---------------------------|------------------------------------------|
| `AZURE_CLIENT_ID`          | Service principal appId (client ID)      |
| `AZURE_TENANT_ID`          | Azure tenant ID                          |
| `AZURE_SUBSCRIPTION_ID`    | Azure subscription ID                    |
| `AZURE_RESOURCE_GROUP`     | e.g. `rg-mlops-demo`                     |
| `AZURE_ML_WORKSPACE`       | e.g. `mlw-mlops-demo`                    |

This project uses **OIDC login** (`azure/login@v2` with `federated-token`) so no client
secret needs to be stored — just configure a federated credential on the App Registration
pointing at your GitHub repo/branch. (If you prefer a classic secret, swap in
`AZURE_CREDENTIALS` and use `creds:` instead — see comment in `cd.yml`.)

## 5. Running locally (without Azure)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data_prep.py --output-dir data/processed
python src/train.py --data-dir data/processed --model-dir outputs/model
python src/evaluate.py --model-dir outputs/model --data-dir data/processed
```

## 6. Running the pipeline on Azure ML

```bash
az ml job create -f pipelines/training_pipeline.yml \
  -g rg-mlops-demo -w mlw-mlops-demo --stream
```

## 7. Deploying

```bash
az ml online-endpoint create -f deployment/endpoint.yml -g rg-mlops-demo -w mlw-mlops-demo
az ml online-deployment create -f deployment/deployment.yml --all-traffic \
  -g rg-mlops-demo -w mlw-mlops-demo
```

## 8. What "MLOps" actually looks like here

| Practice                      | Where |
|--------------------------------|-------|
| Reproducible environments       | `environment/conda.yml` |
| Version-controlled pipeline      | `pipelines/training_pipeline.yml` |
| Automated testing (CI)          | `.github/workflows/ci.yml`, `tests/` |
| Automated training + deployment (CD) | `.github/workflows/cd.yml` |
| Quality gate before deployment    | `src/evaluate.py` (accuracy threshold) |
| Model registry                  | `src/register_model.py` |
| Managed inference endpoint       | `deployment/*.yml`, `src/score.py` |

That's the full loop: code change → CI → pipeline run on Azure → gated evaluation →
model registration → deployment — all from `git push`.
