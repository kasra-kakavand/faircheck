<div align="center">

# 🔍 FairCheck

### Fairness for AI, made simple.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Status: Alpha](https://img.shields.io/badge/status-alpha-orange.svg)]()

**Audit, measure, and mitigate demographic bias in your AI models with one line of code.**

</div>

---

## What is FairCheck?

FairCheck is an open-source Python library that makes fairness analysis in machine learning **simple, accessible, and actionable**. Whether you're a researcher, ML engineer, or AI ethics practitioner, FairCheck helps you understand how your models perform across demographic groups.

## Why FairCheck?

Modern AI systems often exhibit demographic biases that are invisible in aggregate metrics. A model with 95% accuracy might perform dramatically worse for certain groups. FairCheck makes detecting and mitigating these biases as easy as one function call.

```python
from faircheck import audit

results = audit(
    model=your_model,
    dataset=your_data,
    sensitive_attrs=['gender', 'race', 'age']
)

results.report()  # Beautiful interactive dashboard
```

That's it. No more 100-line scripts. No more custom analysis pipelines.

## Key Features

- **One-line auditing** - Comprehensive fairness analysis in a single function call
- **Multiple metrics** - Demographic parity, equalized odds, TPR/FPR disparity, calibration, and more
- **Framework agnostic** - Works with PyTorch, TensorFlow, Scikit-learn, and Hugging Face models
- **Beautiful reports** - Interactive HTML dashboards, PDF exports, and JSON outputs
- **Bias mitigation** - State-of-the-art techniques to reduce bias in trained models
- **Continuous monitoring** - Track fairness in production over time
- **Research-grade** - Implements peer-reviewed methods from leading fairness research

## Installation

> **Note:** FairCheck is currently in active development. The PyPI release is coming soon!

```bash
# Coming soon
pip install faircheck

# Development version
git clone https://github.com/kasra-kakavand/faircheck.git
cd faircheck
pip install -e .
```

## Quick Start

### Basic Audit

```python
import faircheck as fc

# Audit any classifier in one line
results = fc.audit(
    model=my_model,
    dataset=test_data,
    sensitive_attrs=['gender']
)

# View results
print(results.summary())

# Generate detailed report
results.report(output='fairness_report.html')
```

### Bias Mitigation

```python
from faircheck import mitigate

# Apply variance-based fairness regularization
fair_model = mitigate(
    model=biased_model,
    method='variance_regularization',
    train_data=training_data,
    sensitive_attr='skin_tone'
)
```

### Continuous Monitoring

```python
from faircheck import FairnessMonitor

# Set up production monitoring
monitor = FairnessMonitor(model=production_model)

# Track predictions over time
monitor.track(predictions, demographics)

# Get alerts when fairness drifts
if monitor.detect_drift():
    monitor.alert("Fairness threshold violated")
```

## Supported Metrics

FairCheck implements a comprehensive suite of fairness metrics:

| Category | Metrics |
|----------|---------|
| **Group Performance** | Accuracy, TPR, FPR, TNR, FNR per group |
| **Disparity Measures** | Demographic parity, equalized odds, equal opportunity |
| **Statistical** | Disparate impact ratio, statistical parity difference |
| **Calibration** | Calibration error per group, ECE disparity |
| **Custom** | Variance-based regularization, intersectional metrics |

## Why This Matters

> "Fairness in AI is not a feature—it's a requirement."

As AI systems are deployed in healthcare, hiring, criminal justice, and finance, ensuring equitable treatment across demographic groups is critical. FairCheck democratizes fairness analysis, making it accessible to every ML practitioner.

## Roadmap

- [x] Project foundation
- [ ] Core fairness metrics module
- [ ] PyTorch integration
- [ ] Interactive HTML reports
- [ ] PyPI release (v0.1.0)
- [ ] TensorFlow integration
- [ ] Bias mitigation algorithms
- [ ] LLM fairness analysis
- [ ] Production monitoring tools
- [ ] Hugging Face Hub integration
- [ ] Comprehensive documentation site

## Contributing

We welcome contributions! FairCheck is built by the community, for the community. Whether you're fixing bugs, adding features, or improving documentation, your help is appreciated.

```bash
# Fork the repo, then:
git clone https://github.com/YOUR_USERNAME/faircheck.git
cd faircheck
pip install -e ".[dev]"
pytest
```

## Citation

If you use FairCheck in your research, please cite:

```bibtex
@software{kakavand2026faircheck,
  title={FairCheck: A Library for Fairness Auditing in Machine Learning},
  author={Kakavand, Kasra},
  year={2026},
  url={https://github.com/kasra-kakavand/faircheck}
}
```

## License

FairCheck is released under the MIT License. See [LICENSE](LICENSE) for details.

## Author

**Kasra Kakavand**
- GitHub: [@kasra-kakavand](https://github.com/kasra-kakavand)
- Research: [Fairness-Aware Deepfake Detection](https://github.com/kasra-kakavand/deepfake-fairness)

## Acknowledgments

FairCheck builds on decades of research in algorithmic fairness. Special thanks to the contributors of fairness research and the open-source ML community.

---

<div align="center">

**Built with care for a fairer AI future.**

If you find FairCheck useful, please consider giving it a star ⭐

</div>
