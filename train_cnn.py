"""Top-level execution script for training 1D CNN signal models.

Educational Software Simulation.
"""

from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.training.train_deep import DeepLearningTrainer

logger = setup_logger("train_cnn_entry")


def main():
    """Main execution function for 1D CNN signal model training."""
    config = load_config("config.yaml")
    set_seed(config.get("random_seed", 42))

    logger.info("Initializing 1D CNN training script...")
    trainer = DeepLearningTrainer(config)
    model = trainer.train_model("cnn_1d")
    logger.info("1D CNN model architecture initialized successfully.")


if __name__ == "__main__":
    main()
