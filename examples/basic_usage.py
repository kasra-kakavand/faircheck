"""
Basic usage example for FairCheck.

This example demonstrates how to use FairCheck to audit a model's
fairness across demographic groups. We use a simple synthetic example
to show the core API.
"""

import numpy as np

from faircheck import audit


def main():
    """Demonstrate basic FairCheck usage."""

    print("=" * 70)
    print("FairCheck Basic Usage Example")
    print("=" * 70)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Generate synthetic data
    # Simulating a binary classifier with demographic bias
    n_samples = 200

    # Ground truth labels (balanced)
    y_true = np.random.randint(0, 2, n_samples)

    # Demographic attributes
    gender = np.random.choice(["male", "female"], n_samples)
    age_group = np.random.choice(["young", "middle", "senior"], n_samples)

    # Simulate biased predictions:
    # Model is accurate for males but less accurate for females
    y_pred = y_true.copy()

    # Add bias: flip 30% of female predictions
    female_mask = gender == "female"
    flip_indices = np.random.choice(
        np.where(female_mask)[0],
        size=int(0.3 * np.sum(female_mask)),
        replace=False,
    )
    y_pred[flip_indices] = 1 - y_pred[flip_indices]

    print(f"\nDataset Statistics:")
    print(f"  Total samples: {n_samples}")
    print(f"  Gender distribution: {dict(zip(*np.unique(gender, return_counts=True)))}")
    print(f"  Age distribution: {dict(zip(*np.unique(age_group, return_counts=True)))}")

    # Run the audit
    print("\nRunning fairness audit...")

    results = audit(
        y_true=y_true,
        y_pred=y_pred,
        sensitive_attrs={
            "gender": gender,
            "age_group": age_group,
        },
        model_info={
            "name": "Example Classifier",
            "type": "Binary Classification",
        },
    )

    # Print summary
    print("\n" + results.summary())

    # Export to dictionary
    print("\nExporting results to dictionary...")
    results_dict = results.to_dict()
    print(f"Exported keys: {list(results_dict.keys())}")

    # Save report to file
    print("\nSaving report to file...")
    results.report(output="example_audit_report.txt")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
