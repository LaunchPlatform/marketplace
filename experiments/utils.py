import logging

import mlflow

logger = logging.getLogger(__name__)


def ensure_experiment(name: str) -> str:
    try:
        experiment_id = mlflow.create_experiment(
            name=name,
        )
        logger.info("Created experiment with name %s and id %s", name, experiment_id)
        return experiment_id
    except mlflow.exceptions.MlflowException as e:
        logger.info("Failed to create experiment with error: %s", e)
        # If experiment already exists, get its ID
        experiment = mlflow.get_experiment_by_name(name)
        experiment_id = experiment.experiment_id
        logger.info("Return existing experiment id %s for %s", experiment_id, name)
        return experiment_id
