"""Top-level execution script for training classical ML models.

Educational Software Simulation.
"""

import sys
from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.training.train_classical import ClassicalTrainer

logger = setup_logger("train_classical_entry")


def main():
    """Main execution function for classical model training."""

    config = load_config("config.yaml")
    set_seed(config.get("random_seed", 42))

    # TODO: Load processed training dataset, fit models, evaluate, and save artifacts under models/
    logger.info("Initializing classical ML models training script...")
    trainer = ClassicalTrainer(config)
    logger.info("Classical trainer ready. Run data pipeline generation before full training.")


if __name__ == "__main__":
    main()
