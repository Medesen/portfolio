"""
MLflow Model Registry management script.

Provides commands for:
- Promoting models to different stages (Staging, Production, Archived)
- Listing registered models and versions
- Getting model details
- Transitioning model stages
"""

import argparse
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import datetime

import mlflow
from mlflow.tracking import MlflowClient
from tabulate import tabulate

# Model name used in this project
MODEL_NAME = "churn_prediction_model"


def setup_mlflow():
    """Configure MLflow tracking URI (MLFLOW_TRACKING_URI env var, default ./mlruns)."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def list_models(client: MlflowClient):
    """List all registered models."""
    try:
        models = client.search_registered_models()

        if not models:
            print("\nNo models registered yet.")
            print("   Register a model by training with: MLFLOW_REGISTER_MODEL=true make train")
            return

        print("\n" + "=" * 80)
        print("REGISTERED MODELS")
        print("=" * 80)

        for model in models:
            print(f"\nModel: {model.name}")
            print(
                f"   Created: {datetime.fromtimestamp(model.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(
                f"   Last Updated: {datetime.fromtimestamp(model.last_updated_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"   Description: {model.description or 'No description'}")

            # Get latest versions via search_model_versions —
            # get_latest_versions is deprecated in MLflow 2.x (same pattern
            # as src/api/service.py)
            all_versions = client.search_model_versions(f"name='{model.name}'")
            newest_per_stage: dict = {}
            for v in all_versions:
                stage_key = v.current_stage or "None"
                if stage_key not in newest_per_stage or int(v.version) > int(
                    newest_per_stage[stage_key].version
                ):
                    newest_per_stage[stage_key] = v
            latest_versions = list(newest_per_stage.values())
            if latest_versions:
                print("\n   Latest Versions by Stage:")
                table_data = []
                for version in latest_versions:
                    table_data.append(
                        [
                            version.version,
                            version.current_stage,
                            version.run_id[:8],
                            datetime.fromtimestamp(version.creation_timestamp / 1000).strftime(
                                "%Y-%m-%d %H:%M"
                            ),
                        ]
                    )
                print(
                    "   "
                    + tabulate(
                        table_data,
                        headers=["Version", "Stage", "Run ID", "Created"],
                        tablefmt="simple",
                    ).replace("\n", "\n   ")
                )

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nError listing models: {e}")
        sys.exit(1)


def list_versions(client: MlflowClient, model_name: str):
    """List all versions of a registered model."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")

        if not versions:
            print(f"\nNo versions found for model: {model_name}")
            return

        print("\n" + "=" * 80)
        print(f"MODEL VERSIONS: {model_name}")
        print("=" * 80)

        # Sort by version number (descending)
        versions_sorted = sorted(versions, key=lambda v: int(v.version), reverse=True)

        table_data = []
        for version in versions_sorted:
            table_data.append(
                [
                    version.version,
                    version.current_stage,
                    version.run_id[:12],
                    datetime.fromtimestamp(version.creation_timestamp / 1000).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    version.description[:40] if version.description else "",
                ]
            )

        print(
            tabulate(
                table_data,
                headers=["Version", "Stage", "Run ID", "Created", "Description"],
                tablefmt="grid",
            )
        )

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nError listing versions: {e}")
        sys.exit(1)


def promote_model(client: MlflowClient, model_name: str, version: str, stage: str):
    """
    Promote a model version to a specific stage.

    Args:
        client: MLflow client
        model_name: Name of the registered model
        version: Version number to promote
        stage: Target stage (Staging, Production, Archived)
    """
    valid_stages = ["Staging", "Production", "Archived", "None"]

    if stage not in valid_stages:
        print(f"\nInvalid stage: {stage}")
        print(f"   Valid stages: {', '.join(valid_stages)}")
        sys.exit(1)

    try:
        # Get current version info
        version_info = client.get_model_version(model_name, version)
        current_stage = version_info.current_stage

        print("\nPromoting model...")
        print(f"   Model: {model_name}")
        print(f"   Version: {version}")
        print(f"   Current Stage: {current_stage}")
        print(f"   Target Stage: {stage}")

        # Transition model version stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage,
            archive_existing_versions=(stage == "Production"),  # Archive old prod models
        )

        print(f"\nSuccessfully promoted model version {version} to {stage}")

        # Show what happened to other versions if promoting to Production
        if stage == "Production":
            print("\nNote: Existing Production models were archived automatically")

        # Show current state
        print("\nCurrent state:")
        list_versions(client, model_name)

    except Exception as e:
        print(f"\nError promoting model: {e}")
        sys.exit(1)


def get_model_info(client: MlflowClient, model_name: str, version: str = None, stage: str = None):
    """Get detailed information about a model version."""
    try:
        if version:
            model_version = client.get_model_version(model_name, version)
            versions = [model_version]
            title = f"MODEL INFO: {model_name} v{version}"
        elif stage:
            # search_model_versions instead of the deprecated
            # get_latest_versions(stages=...) — same pattern as src/api/service.py
            staged = [
                v
                for v in client.search_model_versions(f"name='{model_name}'")
                if v.current_stage == stage
            ]
            versions = [max(staged, key=lambda v: int(v.version))] if staged else []
            if not versions:
                print(f"\nNo model in stage '{stage}'")
                return
            title = f"MODEL INFO: {model_name} ({stage} stage)"
        else:
            print("\nMust specify either --version or --stage")
            sys.exit(1)

        print("\n" + "=" * 80)
        print(title)
        print("=" * 80)

        for mv in versions:
            print(f"\nVersion: {mv.version}")
            print(f"   Stage: {mv.current_stage}")
            print(f"   Run ID: {mv.run_id}")
            print(f"   Source: {mv.source}")
            print(
                f"   Created: {datetime.fromtimestamp(mv.creation_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(
                f"   Updated: {datetime.fromtimestamp(mv.last_updated_timestamp/1000).strftime('%Y-%m-%d %H:%M:%S')}"
            )
            print(f"   Description: {mv.description or 'No description'}")
            print(f"   Status: {mv.status}")

            # Get run info for additional details
            try:
                run = client.get_run(mv.run_id)
                print("\n   Metrics from Training Run:")
                for key, value in sorted(run.data.metrics.items()):
                    if isinstance(value, float):
                        print(f"      {key}: {value:.4f}")
                    else:
                        print(f"      {key}: {value}")
            except Exception:
                pass

        print("\n" + "=" * 80)

    except Exception as e:
        print(f"\nError getting model info: {e}")
        sys.exit(1)


def delete_version(client: MlflowClient, model_name: str, version: str):
    """Delete a specific model version."""
    try:
        # Get version info first
        version_info = client.get_model_version(model_name, version)

        print("\n WARNING: About to delete model version")
        print(f"   Model: {model_name}")
        print(f"   Version: {version}")
        print(f"   Stage: {version_info.current_stage}")

        # Confirm deletion
        response = input("\n   Type 'yes' to confirm deletion: ")
        if response.lower() != "yes":
            print("\nDeletion cancelled")
            return

        # Delete the version
        client.delete_model_version(model_name, version)
        print(f"\nSuccessfully deleted version {version}")

    except Exception as e:
        print(f"\nError deleting version: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Manage MLflow Model Registry",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all registered models
  python scripts/manage_registry.py list
  
  # List all versions of the churn model
  python scripts/manage_registry.py versions
  
  # Promote version 2 to Production
  python scripts/manage_registry.py promote --version 2 --stage Production
  
  # Get info about Production model
  python scripts/manage_registry.py info --stage Production
  
  # Get info about specific version
  python scripts/manage_registry.py info --version 3
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # List models command
    subparsers.add_parser("list", help="List all registered models")

    # List versions command
    versions_parser = subparsers.add_parser("versions", help="List all versions of the model")
    versions_parser.add_argument(
        "--name", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})"
    )

    # Promote command
    promote_parser = subparsers.add_parser("promote", help="Promote a model version to a stage")
    promote_parser.add_argument(
        "--name", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})"
    )
    promote_parser.add_argument("--version", required=True, help="Version number to promote")
    promote_parser.add_argument(
        "--stage",
        required=True,
        choices=["Staging", "Production", "Archived", "None"],
        help="Target stage",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Get detailed model information")
    info_parser.add_argument(
        "--name", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})"
    )
    group = info_parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--version", help="Specific version number")
    group.add_argument(
        "--stage",
        choices=["Staging", "Production", "Archived", "None"],
        help="Get model in specific stage",
    )

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete a model version")
    delete_parser.add_argument(
        "--name", default=MODEL_NAME, help=f"Model name (default: {MODEL_NAME})"
    )
    delete_parser.add_argument("--version", required=True, help="Version number to delete")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup MLflow client
    client = setup_mlflow()

    # Execute command
    if args.command == "list":
        list_models(client)
    elif args.command == "versions":
        list_versions(client, args.name)
    elif args.command == "promote":
        promote_model(client, args.name, args.version, args.stage)
    elif args.command == "info":
        get_model_info(client, args.name, args.version, args.stage)
    elif args.command == "delete":
        delete_version(client, args.name, args.version)


if __name__ == "__main__":
    main()
