"""Register a trained model into the Azure ML model registry, gated on metrics.json.

Usage:
    python src/register_model.py \
        --model-dir outputs/model \
        --model-name iris-classifier \
        --subscription-id <sub> --resource-group <rg> --workspace-name <ws>
"""
import argparse
import json
import os

from azure.ai.ml import MLClient
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.identity import DefaultAzureCredential


def register(
    model_dir: str,
    model_name: str,
    subscription_id: str,
    resource_group: str,
    workspace_name: str,
    min_accuracy: float = 0.85,
) -> None:
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path) as f:
        metrics = json.load(f)

    if metrics["accuracy"] < min_accuracy:
        raise RuntimeError(
            f"Refusing to register: accuracy {metrics['accuracy']} below {min_accuracy}"
        )

    credential = DefaultAzureCredential()
    ml_client = MLClient(credential, subscription_id, resource_group, workspace_name)

    model = Model(
        path=model_dir,
        name=model_name,
        type=AssetTypes.CUSTOM_MODEL,
        description="RandomForest Iris classifier, registered via CI/CD.",
        properties=metrics,
    )

    registered = ml_client.models.create_or_update(model)
    print(f"Registered model: {registered.name}, version: {registered.version}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--subscription-id", type=str, required=True)
    parser.add_argument("--resource-group", type=str, required=True)
    parser.add_argument("--workspace-name", type=str, required=True)
    parser.add_argument("--min-accuracy", type=float, default=0.85)
    args = parser.parse_args()

    register(
        args.model_dir,
        args.model_name,
        args.subscription_id,
        args.resource_group,
        args.workspace_name,
        args.min_accuracy,
    )


if __name__ == "__main__":
    main()
