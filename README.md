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
```

Step 4 — setting up GitHub Actions authentication (a service principal with an OIDC
federated credential) — is covered just below, since it has a few gotchas worth walking
through carefully.

This project uses **OIDC login** (`azure/login@v2`) so no client secret needs to be stored.
This requires a **federated credential** on the App Registration, set up like this:

```bash
# 1. Create the app registration + service principal
APP_ID=$(az ad app create --display-name "sp-mlops-github" --query appId -o tsv)
az ad sp create --id "$APP_ID"

# 2. Grant it Contributor on the resource group
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
az role assignment create --assignee "$APP_ID" --role Contributor \
  --scope "/subscriptions/$SUBSCRIPTION_ID/resourceGroups/rg-mlops-demo"

# 3. Add the federated credential — this is the part that's easy to get wrong (see note below)
az ad app federated-credential create --id "$APP_ID" --parameters '{
  "name": "github-mlops-main-branch",
  "issuer": "https://token.actions.githubusercontent.com",
  "subject": "repo:<owner>/<repo>:ref:refs/heads/main",
  "audiences": ["api://AzureADTokenExchange"]
}'

# 4. Get the tenant ID
az account show --query tenantId -o tsv
```

**Important — the `subject` must match exactly what GitHub sends.** If your GitHub account
or repo was ever renamed, GitHub's OIDC subject claim includes the immutable numeric ID
alongside the current name, e.g. `repo:owner@123456/repo@789012:ref:refs/heads/main`
instead of the plain `repo:owner/repo:ref:refs/heads/main` you'd expect. Don't guess this —
trigger the workflow once (it will fail at login), then copy the **exact** `subject claim`
line from the failed run's log and use that verbatim in step 3 above.

If login still fails with `AADSTS70025: ... has no configured federated identity credentials`
after creating it, verify:
```bash
az ad app federated-credential list --id "$APP_ID" -o table   # confirm subject matches exactly
az ad app show --id "$APP_ID" --query appId -o tsv             # confirm this matches your AZURE_CLIENT_ID secret
```
Both the app ID and the subject string have to match precisely — a mismatch on either one
produces the same error.

Then in your GitHub repo, add these **secrets** (Settings → Secrets and variables → Actions):

| Secret name              | Value                                   |
|---------------------------|------------------------------------------|
| `AZURE_CLIENT_ID`          | Service principal appId (client ID)      |
| `AZURE_TENANT_ID`          | Azure tenant ID                          |
| `AZURE_SUBSCRIPTION_ID`    | Azure subscription ID                    |
| `AZURE_RESOURCE_GROUP`     | e.g. `rg-mlops-demo`                     |
| `AZURE_ML_WORKSPACE`       | e.g. `mlw-mlops-demo`                    |

(If you prefer a classic secret instead of OIDC, swap in `AZURE_CREDENTIALS` and use
`creds:` instead — see the commented-out alternative in `cd.yml`.)

## 5. Running locally (without Azure)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python src/data_prep.py --output-dir data/processed
python src/train.py --data-dir data/processed --model-dir outputs/model
python src/evaluate.py --model-dir outputs/model --data-dir data/processed --output-dir outputs/metrics
```

`evaluate.py` writes `metrics.json` to its own `--output-dir`, separate from `--model-dir`.
This matters more than it looks — see the note in Section 6 about read-only inputs on Azure ML.

## 6. Running the pipeline on Azure ML

Run this from **your own terminal** (Azure Cloud Shell, or a local terminal with the Azure
CLI installed and `az login` done) — not something that runs inside the repo's own code.
Run it from the **project root** (not from inside `pipelines/`), since the YAML uses
relative paths like `code: ../` to find `src/`.

Prerequisites (one-time, see Section 4):
- The `ml` CLI extension is installed: `az extension add -n ml`
- The resource group and workspace exist
- The `cpu-cluster` compute target exists (referenced by name in `training_pipeline.yml`)

```bash
az ml job create -f pipelines/training_pipeline.yml \
  -g rg-mlops-demo -w mlw-mlops-demo --stream
```

First run is slow: Azure ML has to provision a compute node from zero (since
`min_instances: 0`) and build the Docker image from `environment/conda.yml` before any
step logs appear — this alone can take 5–10+ minutes. A healthy run ends with
`PASSED quality gate: ...` in the `evaluate_job` logs and overall status `Completed`.

**Why `evaluate_job` writes metrics to its own output folder:** Azure ML mounts a step's
*inputs* (data coming from a previous step in the pipeline) as **read-only**. Writing
`metrics.json` into `model_dir` (which is `train_job`'s output, consumed as `evaluate_job`'s
input) fails with `OSError: [Errno 30] Read-only file system`. The fix is what's already
in this repo: `evaluate_job` has its own `metrics_output` (`uri_folder`), and `evaluate.py`
takes a separate `--output-dir` for that file instead of reusing `--model-dir`.

**Why `model_output` and `metrics_output` are declared at the pipeline level, not just on
`train_job`/`evaluate_job`:** `az ml job download` (used by the CD workflow to pull the
trained model back down for registration) only fetches what's registered under the
**pipeline job's own** `outputs:` section. Outputs declared only on child steps aren't
visible to a plain download of the parent job — you'd get an empty `named-outputs/`
folder and a confusing `FileNotFoundError` when the registration script tries to read
`metrics.json`. The fix (already in this repo): the pipeline declares `model_output` and
`metrics_output` at the top level, and `train_job`/`evaluate_job` bind their own outputs
to `${{parent.outputs.model_output}}` / `${{parent.outputs.metrics_output}}`.

**Common errors and fixes here:**
- `Not found compute with name cpu-cluster` → the compute cluster hasn't been created yet;
  run the `az ml compute create ...` command from Section 4.
- Quota / VM size errors on `Standard_DS3_v2` → try `Standard_DS2_v2`, or check
  `az vm list-usage -l eastus -o table` for what your subscription actually has quota for.

## 7. Deploying

The pipeline in Section 6 only trains and evaluates the model — it does **not** register
or deploy it. That's a separate, deliberate step (mirroring what the CD workflow automates):

**a) Register the model** (needed before the deployment step below will work — the
`ModelNotFound` error happens if you skip this):

```bash
# Quickest path: register directly from a local training run
python src/train.py --data-dir data/processed --model-dir outputs/model
az ml model create --name iris-classifier --version 1 \
  --path outputs/model -g rg-mlops-demo -w mlw-mlops-demo

# Or, register the model produced by an actual Azure ML pipeline job:
az ml job download -n <job-name> --download-path ./pipeline_outputs \
  -g rg-mlops-demo -w mlw-mlops-demo
az ml model create --name iris-classifier --version 1 \
  --path ./pipeline_outputs/named-outputs/model_output \
  -g rg-mlops-demo -w mlw-mlops-demo
```

Check what's registered any time with `az ml model list -g rg-mlops-demo -w mlw-mlops-demo -o table`.

**b) Register the required resource providers** — a one-time step per subscription.
If you hit `(SubscriptionNotRegistered)` on the endpoint step below, run:

```bash
az provider register --namespace Microsoft.MachineLearningServices
az provider register --namespace Microsoft.ContainerRegistry
az provider register --namespace Microsoft.PolicyInsights
az provider register --namespace Microsoft.Network
az provider register --namespace Microsoft.Storage
az provider register --namespace Microsoft.KeyVault
az provider register --namespace Microsoft.Insights
```

Wait for `Registered` status (1–5 min): `az provider show --namespace Microsoft.MachineLearningServices --query registrationState -o tsv`

**c) Create the endpoint and deployment:**

```bash
az ml online-endpoint create -f deployment/endpoint.yml -g rg-mlops-demo -w mlw-mlops-demo
az ml online-deployment create -f deployment/deployment.yml --all-traffic \
  -g rg-mlops-demo -w mlw-mlops-demo
```

**Common errors and fixes here:**
- `An endpoint with this name already exists` → harmless if it's healthy; check its state
  with `az ml online-endpoint show -n iris-classifier-endpoint -g rg-mlops-demo -w mlw-mlops-demo`.
  If `provisioning_state` is `Failed` (often a leftover from a `SubscriptionNotRegistered`
  error above), delete and recreate it:
  ```bash
  az ml online-endpoint delete -n iris-classifier-endpoint -g rg-mlops-demo -w mlw-mlops-demo -y
  az ml online-endpoint create -f deployment/endpoint.yml -g rg-mlops-demo -w mlw-mlops-demo
  ```
- `ModelNotFound: Model container with name: iris-classifier not found` → the model
  hasn't been registered yet; do step (a) above first.
- `Instance type Standard_DS2_v2 may be too small...` → this is just an advisory warning,
  safe to ignore for a demo-scale model like this one.

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
