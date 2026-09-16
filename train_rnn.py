"""Top-level execution script for training RNN sequence models.

Educational Software Simulation.
"""

from src.utils.config_loader import load_config
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.training.train_deep import DeepLearningTrainer

logger = setup_logger("train_rnn_entry")


def main():
    """Main execution function for RNN temporal model training."""
    config = load_config("config.yaml")
    set_seed(config.get("random_seed", 42))

    logger.info("Initializing RNN sequence model training script...")
    trainer = DeepLearningTrainer(config)
    model = trainer.train_model("rnn")
    logger.info("RNN sequence model architecture initialized successfully.")


if __name__ == "__main__":
    main()
