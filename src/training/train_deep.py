"""Deep Learning Training Pipeline.

Educational Simulation System.
Implements full PyTorch training loops with DataLoaders, optimizer, cross-entropy
loss, epoch-level train/val tracking, and best-model checkpointing for:
  - 1D CNN  (signal/tabular features)
  - RNN     (temporal sequences)
  - LSTM    (temporal sequences)
"""

from typing import Dict, Any, Optional, List, Tuple
import os
import json
import random
import time

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

import numpy as np

from src.models.deep import Physio1DCNN, PhysioRNN, PhysioLSTM
from src.evaluation.metrics import ModelEvaluator
from src.utils.logger import setup_logger

logger = setup_logger("train_deep")

# Stable per-architecture offsets so each model is reproducible without all
# three sharing the same RNG stream (and without depending on call order).
MODEL_SEED_OFFSETS = {"cnn_1d": 0, "rnn": 1000, "lstm": 2000}


class DeepLearningTrainer:
    """Trainer orchestrator for PyTorch deep learning models."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize deep learning trainer.

        Args:
            config: Pipeline configuration dictionary.
        """
        self.config = config
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not installed. Run: pip install torch")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.evaluator = ModelEvaluator(
            target_names=config.get("target", ["NORMAL", "MODERATE", "HIGH", "CRITICAL"])
        )
        self.models_dir = os.path.join(config.get("paths", {}).get("models", "models"), "deep")
        self.results_dir = os.path.join(
            config.get("paths", {}).get("experiments", "experiments"), "results"
        )
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        logger.info(f"Initialized DeepLearningTrainer | device={self.device}")

    # ------------------------------------------------------------------
    # DataLoader helpers
    # ------------------------------------------------------------------

    def _make_loader(
        self, X: np.ndarray, y: np.ndarray, batch_size: int,
        shuffle: bool = True, seed: Optional[int] = None,
    ) -> "DataLoader":
        """Build a PyTorch DataLoader from numpy arrays.

        Args:
            X: Feature array.
            y: Label array.
            batch_size: Mini-batch size.
            shuffle: Whether to shuffle samples.
            seed: Optional generator seed for deterministic shuffling.

        Returns:
            DataLoader instance.
        """
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_t, y_t)
        generator = None
        if shuffle and seed is not None:
            generator = torch.Generator()
            generator.manual_seed(int(seed))
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=0, generator=generator,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_epoch(
        self, model: "nn.Module", loader: "DataLoader",
        optimizer: "torch.optim.Optimizer", criterion: "nn.Module"
    ) -> Tuple[float, float]:
        """Run one training epoch.

        Args:
            model: PyTorch model.
            loader: Training DataLoader.
            optimizer: Optimizer instance.
            criterion: Loss function.

        Returns:
            Tuple of (avg_loss, accuracy) over all batches.
        """
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y_batch).sum().item()
            total += X_batch.size(0)

        return total_loss / total, correct / total

    def _eval_epoch(
        self, model: "nn.Module", loader: "DataLoader", criterion: "nn.Module"
    ) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """Evaluate model on a DataLoader.

        Args:
            model: PyTorch model.
            loader: DataLoader.
            criterion: Loss function.

        Returns:
            Tuple of (avg_loss, accuracy, y_true, y_pred).
        """
        model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_true = [], []

        with torch.no_grad():
            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                total_loss += loss.item() * X_batch.size(0)
                preds = logits.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += X_batch.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(y_batch.cpu().numpy())

        return (
            total_loss / total,
            correct / total,
            np.array(all_true),
            np.array(all_preds),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train_model(
        self,
        model_type: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Train a specified deep learning model with a full training loop.

        Args:
            model_type: Architecture ('cnn_1d', 'rnn', 'lstm').
            X_train: Training data array.
            y_train: Training integer labels.
            X_val:   Validation data array (optional).
            y_val:   Validation integer labels (optional).
            X_test:  Test data array (optional).
            y_test:  Test integer labels (optional).

        Returns:
            Dict with keys: 'model', 'history', 'val_metrics', 'test_metrics'.
        """
        deep_cfg = self.config.get("models", {}).get("deep", {})

        # Seed before construction so weight init, dropout and shuffling are all
        # deterministic. Previously nothing was seeded, so "reproducible" reruns
        # produced different deep metrics on byte-identical signal data.
        base_seed = int(self.config.get("random_seed", 42))
        model_seed = base_seed + MODEL_SEED_OFFSETS.get(model_type, 0)
        random.seed(model_seed)
        np.random.seed(model_seed)
        torch.manual_seed(model_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(model_seed)

        # ── Build model ───────────────────────────────────────────────
        if model_type == "cnn_1d":
            cfg = deep_cfg.get("cnn_1d", {})
            model = Physio1DCNN(
                in_channels=cfg.get("num_channels", 1),
                num_classes=cfg.get("num_classes", 4),
                kernel_size=cfg.get("kernel_size", 5),
            )
            lr         = cfg.get("learning_rate", 0.001)
            epochs     = cfg.get("epochs", 20)
            batch_size = cfg.get("batch_size", 64)

        elif model_type == "rnn":
            cfg = deep_cfg.get("rnn", {})
            model = PhysioRNN(
                input_size=cfg.get("input_size", 6),
                hidden_size=cfg.get("hidden_size", 64),
                num_layers=cfg.get("num_layers", 2),
                num_classes=cfg.get("num_classes", 4),
            )
            lr         = cfg.get("learning_rate", 0.001)
            epochs     = cfg.get("epochs", 20)
            batch_size = cfg.get("batch_size", 64)

        elif model_type == "lstm":
            cfg = deep_cfg.get("lstm", {})
            model = PhysioLSTM(
                input_size=cfg.get("input_size", 6),
                hidden_size=cfg.get("hidden_size", 64),
                num_layers=cfg.get("num_layers", 2),
                num_classes=cfg.get("num_classes", 4),
            )
            lr         = cfg.get("learning_rate", 0.001)
            epochs     = cfg.get("epochs", 20)
            batch_size = cfg.get("batch_size", 64)

        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                             f"Choose from ['cnn_1d', 'rnn', 'lstm']")

        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        train_loader = self._make_loader(
            X_train, y_train, batch_size, shuffle=True, seed=model_seed
        )
        val_loader   = self._make_loader(X_val,   y_val,   batch_size, shuffle=False) if X_val is not None else None
        test_loader  = self._make_loader(X_test,  y_test,  batch_size, shuffle=False) if X_test is not None else None

        logger.info("=" * 60)
        logger.info(f"Training [{model_type.upper()}] | epochs={epochs}, lr={lr}, batch={batch_size}")
        logger.info(f"  X_train={X_train.shape}, device={self.device}")
        logger.info("=" * 60)

        # ── Epoch loop ────────────────────────────────────────────────
        history: Dict[str, List[float]] = {
            "train_loss": [], "train_acc": [],
            "val_loss":   [], "val_acc":   [],
        }
        best_val_acc   = -1.0
        best_ckpt_path = os.path.join(self.models_dir, f"{model_type}_best.pt")
        checkpoint_saved = False

        t_total = time.time()
        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(model, train_loader, optimizer, criterion)
            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc,  4))

            val_str = ""
            if val_loader is not None:
                val_loss, val_acc, _, _ = self._eval_epoch(model, val_loader, criterion)
                history["val_loss"].append(round(val_loss, 4))
                history["val_acc"].append(round(val_acc,  4))
                val_str = f"  val_loss={val_loss:.4f}  val_acc={val_acc:.4f}"

                # Save best checkpoint
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    torch.save(model.state_dict(), best_ckpt_path)
                    checkpoint_saved = True

            epoch_time = round(time.time() - t0, 1)
            logger.info(
                f"  Epoch {epoch:>3}/{epochs}  "
                f"train_loss={train_loss:.4f}  train_acc={train_acc:.4f}"
                f"{val_str}  [{epoch_time}s]"
            )

        total_time = round(time.time() - t_total, 1)
        if val_loader is not None:
            logger.info(f"Training complete in {total_time}s. Best val acc={best_val_acc:.4f}")
        else:
            logger.info(f"Training complete in {total_time}s. No validation split provided.")

        # ── Load best checkpoint ──────────────────────────────────────
        # Only load a file this run wrote. Without a validation loader nothing is
        # saved, and blindly loading `*_best.pt` would resurrect a stale model
        # from a previous run (or a previous architecture) for the final metrics.
        if checkpoint_saved:
            model.load_state_dict(torch.load(best_ckpt_path, map_location=self.device))
            logger.info(f"Loaded best checkpoint from {best_ckpt_path}")
        elif val_loader is None:
            torch.save(model.state_dict(), best_ckpt_path)
            checkpoint_saved = True
            logger.info(f"No validation set: saved final-epoch checkpoint to {best_ckpt_path}")
        else:
            logger.warning("No checkpoint was saved during training; keeping final-epoch weights.")

        # ── Final evaluation ──────────────────────────────────────────
        val_metrics = test_metrics = None

        if val_loader is not None:
            _, _, y_true_val, y_pred_val = self._eval_epoch(model, val_loader, criterion)
            val_metrics = self.evaluator.evaluate(y_true_val, y_pred_val)
            logger.info(
                f"  Final Val   Accuracy={val_metrics['accuracy']:.4f}  "
                f"MacroF1={val_metrics['macro_f1']:.4f}"
            )

        if test_loader is not None:
            _, _, y_true_test, y_pred_test = self._eval_epoch(model, test_loader, criterion)
            test_metrics = self.evaluator.evaluate(y_true_test, y_pred_test)
            logger.info(
                f"  Final Test  Accuracy={test_metrics['accuracy']:.4f}  "
                f"MacroF1={test_metrics['macro_f1']:.4f}"
            )

        return {
            "model":        model,
            "history":      history,
            "val_metrics":  val_metrics,
            "test_metrics": test_metrics,
            "model_path":   best_ckpt_path,
            "total_time_s": total_time,
        }

    def train_all(
        self,
        X_temporal_train: np.ndarray, y_temporal_train: np.ndarray,
        X_temporal_val:   np.ndarray, y_temporal_val:   np.ndarray,
        X_temporal_test:  np.ndarray, y_temporal_test:  np.ndarray,
        X_signal_train:   np.ndarray, y_signal_train:   np.ndarray,
        X_signal_val:     np.ndarray, y_signal_val:     np.ndarray,
        X_signal_test:    np.ndarray, y_signal_test:    np.ndarray,
    ) -> Dict[str, Any]:
        """Train all deep learning models and save a results summary.

        Args:
            X_temporal_* / y_temporal_*: Temporal sequence splits (for RNN, LSTM).
            X_signal_*   / y_signal_*:   Signal splits (for CNN 1D).

        Returns:
            Dict mapping model_type to training result dicts.
        """
        results = {}

        # 1D CNN on signal data
        results["cnn_1d"] = self.train_model(
            "cnn_1d",
            X_signal_train,   y_signal_train,
            X_signal_val,     y_signal_val,
            X_signal_test,    y_signal_test,
        )

        # RNN on temporal data
        results["rnn"] = self.train_model(
            "rnn",
            X_temporal_train, y_temporal_train,
            X_temporal_val,   y_temporal_val,
            X_temporal_test,  y_temporal_test,
        )

        # LSTM on temporal data
        results["lstm"] = self.train_model(
            "lstm",
            X_temporal_train, y_temporal_train,
            X_temporal_val,   y_temporal_val,
            X_temporal_test,  y_temporal_test,
        )

        # ── Save results JSON ─────────────────────────────────────────
        summary = {}
        for name, r in results.items():
            vm = r["val_metrics"]  or {}
            tm = r["test_metrics"] or {}
            summary[name] = {
                "val_accuracy":   round(vm.get("accuracy", 0), 4),
                "val_macro_f1":   round(vm.get("macro_f1", 0), 4),
                "test_accuracy":  round(tm.get("accuracy", 0), 4),
                "test_macro_f1":  round(tm.get("macro_f1", 0), 4),
                "total_time_s":   r["total_time_s"],
                "model_path":     r["model_path"],
                "history":        r["history"],
            }

        results_path = os.path.join(self.results_dir, "deep_results.json")
        with open(results_path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Deep learning results saved → {results_path}")

        # ── Print summary table ───────────────────────────────────────
        logger.info("\n" + "=" * 60)
        logger.info("DEEP LEARNING — RESULTS SUMMARY")
        logger.info("=" * 60)
        header = f"{'Model':<12} {'Val Acc':>9} {'Val F1':>9} {'Test Acc':>10} {'Test F1':>9}"
        logger.info(header)
        logger.info("-" * 55)
        for name, s in summary.items():
            logger.info(
                f"{name:<12} {s['val_accuracy']:>9.4f} {s['val_macro_f1']:>9.4f} "
                f"{s['test_accuracy']:>10.4f} {s['test_macro_f1']:>9.4f}"
            )
        logger.info("=" * 60)

        return results
