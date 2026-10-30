import os
import logging

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException


# ============================================================
# Configuration
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://localhost:5000"
)

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "XGBoost_Risk_Scorer"
)

MODEL_ALIAS = os.getenv(
    "MODEL_ALIAS",
    "champion"
)

REQUIRE_VALIDATION = (
    os.getenv("REQUIRE_VALIDATION", "false").lower()
    == "true"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ============================================================
# Retrieve Latest Registered Model
# ============================================================

def get_latest_model_version(
    client: MlflowClient,
    model_name: str
):

    versions = client.search_model_versions(
        filter_string=f"name='{model_name}'",
        order_by=["version_number DESC"],
        max_results=1
    )

    if not versions:
        raise ValueError(
            f"No registered versions found for '{model_name}'."
        )

    return versions[0]


# ============================================================
# Champion Promotion
# ============================================================

def promote_latest_to_champion() -> str:

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    client = MlflowClient()

    logger.info(
        "Checking registered model: %s",
        MODEL_NAME
    )

    # --------------------------------------------------------
    # 1. Retrieve latest model version
    # --------------------------------------------------------

    latest_model = get_latest_model_version(
        client,
        MODEL_NAME
    )

    latest_version = str(
        latest_model.version
    )

    logger.info(
        "Latest registered version: %s",
        latest_version
    )

    # --------------------------------------------------------
    # 2. Validate model status
    # --------------------------------------------------------

    if latest_model.status != "READY":
        raise RuntimeError(
            f"Model version {latest_version} "
            f"is not ready. Status: {latest_model.status}"
        )

    # Optional validation gate
    if REQUIRE_VALIDATION:

        validation_status = latest_model.tags.get(
            "validation_status"
        )

        if validation_status != "passed":
            raise RuntimeError(
                f"Model version {latest_version} "
                "has not passed validation."
            )

    # --------------------------------------------------------
    # 3. Check current champion
    # --------------------------------------------------------

    registered_model = client.get_registered_model(
        MODEL_NAME
    )

    current_champion = registered_model.aliases.get(
        MODEL_ALIAS
    )

    if current_champion == latest_version:

        logger.info(
            "Version %s is already the champion.",
            latest_version
        )

        return latest_version

    # --------------------------------------------------------
    # 4. Promote latest model
    # --------------------------------------------------------

    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=MODEL_ALIAS,
        version=latest_version
    )

    logger.info(
        "Promoted model '%s' version %s to '%s'.",
        MODEL_NAME,
        latest_version,
        MODEL_ALIAS
    )

    # --------------------------------------------------------
    # 5. Verify promotion
    # --------------------------------------------------------

    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        MODEL_ALIAS
    )

    if str(champion.version) != latest_version:
        raise RuntimeError(
            "Champion alias verification failed."
        )

    logger.info(
        "Champion verification successful: version %s",
        champion.version
    )

    return latest_version


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":

    try:
        promoted_version = promote_latest_to_champion()

        logger.info(
            "Promotion completed: %s@%s -> version %s",
            MODEL_NAME,
            MODEL_ALIAS,
            promoted_version
        )

    except (MlflowException, ValueError, RuntimeError):
        logger.exception(
            "Champion promotion failed."
        )
        raise