"""Training script for the LTO tradeoff model.

Usage:
    python -m lto.ml.train --samples 10000 --output models/tradeoff_v1.pkl
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train the LTO tradeoff model")
    parser.add_argument(
        "--samples", type=int, default=10000,
        help="Number of synthetic samples to generate (default: 10000)",
    )
    parser.add_argument(
        "--output", type=str, default="models/tradeoff_v1",
        help="Output model name (without .pkl extension)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    args = parser.parse_args()

    # Import here to avoid circular imports
    from lto.ml.engine import TradeoffModel

    logger.info(f"=== LTO Model Training ===")
    logger.info(f"Samples: {args.samples}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Output: {args.output}")

    # Train
    model = TradeoffModel(model_dir="models")
    metrics = model.train_from_simulator(n_samples=args.samples, seed=args.seed)

    # Report
    logger.info("=== Training Results ===")
    for name, rmse in metrics.items():
        logger.info(f"  {name:<30s} RMSE: {rmse:.4f}")

    # Save
    path = model.save(name=args.output.split("/")[-1])
    logger.info(f"Model saved to: {path}")

    # Quick sanity check
    from lto.schemas import JobParameters

    test_params = JobParameters(na=0.33, dose_mj_cm2=15.0)
    prediction = model.predict(test_params)
    logger.info("=== Sanity Check Prediction ===")
    for name, score_ci in prediction.predictions.items():
        logger.info(f"  {name:<30s} {score_ci.score:.3f} [{score_ci.ci_low:.3f}, {score_ci.ci_high:.3f}]")
    logger.info(f"  Uncertainty: {prediction.uncertainty.confidence_level.value}")
    logger.info(f"  Constraints satisfied: {prediction.constraints.all_satisfied}")
    logger.info(f"  Inference time: {prediction.inference_time_ms:.1f}ms")


if __name__ == "__main__":
    main()
