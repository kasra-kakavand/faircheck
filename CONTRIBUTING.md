# Contributing to FairCheck

First off, thank you for considering contributing to FairCheck! It's people like you that make FairCheck a great tool for the AI community.

## Code of Conduct

This project adheres to a code of conduct that we expect all contributors to follow. Please be respectful, inclusive, and constructive in all interactions.

## How Can I Contribute?

### Reporting Bugs

Before creating bug reports, please check existing issues to avoid duplicates. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your Python version, OS, and FairCheck version
- Any relevant code snippets or error messages

### Suggesting Enhancements

Enhancement suggestions are tracked as GitHub issues. When suggesting features:

- Use a clear and descriptive title
- Provide a detailed description of the proposed feature
- Explain why this enhancement would be useful
- Provide examples of how it would be used

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Format your code (`black faircheck/ tests/`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to your fork (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git

### Setting Up Your Development Environment

```bash
# Fork and clone the repository
git clone https://github.com/YOUR_USERNAME/faircheck.git
cd faircheck

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[dev,viz]"

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=faircheck

# Run specific test file
pytest tests/test_metrics.py

# Run with verbose output
pytest -v
```

### Code Style

We use the following tools to maintain code quality:

- **Black** for code formatting
- **Ruff** for linting
- **MyPy** for type checking

```bash
# Format code
black faircheck/ tests/ examples/

# Run linter
ruff check faircheck/ tests/ examples/

# Type check
mypy faircheck/
```

## Project Structure

faircheck/
├── faircheck/          # Main package
│   ├── core/          # Core functionality
│   ├── mitigation/    # Bias mitigation (planned)
│   └── visualization/ # Reports and plots (planned)
├── tests/             # Test suite
├── examples/          # Usage examples
└── docs/             # Documentation

## Coding Standards

### Python Style

- Follow PEP 8
- Use type hints for all public APIs
- Write descriptive variable names
- Add docstrings to all public functions and classes (NumPy style)
- Maximum line length: 100 characters

### Docstring Example

```python
def calculate_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_attr: np.ndarray,
) -> float:
    """
    Calculate a fairness metric.

    Args:
        y_true: Ground truth labels of shape (n,)
        y_pred: Predicted labels of shape (n,)
        sensitive_attr: Demographic groups of shape (n,)

    Returns:
        Fairness metric value between 0 and 1

    Example:
        >>> metric = calculate_metric(y_true, y_pred, groups)
        >>> print(f"Metric: {metric:.3f}")
    """
    pass
```

### Testing Standards

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use descriptive test names: `test_what_should_happen_when_condition`
- Include edge cases and error conditions

## Commit Messages

Use clear, descriptive commit messages following this format:
<type>: <description>
[optional body]
[optional footer]

### Types

- **feat**: New feature
- **fix**: Bug fix
- **docs**: Documentation only changes
- **style**: Code style changes (formatting, etc.)
- **refactor**: Code refactoring
- **test**: Adding or updating tests
- **chore**: Maintenance tasks

### Examples

feat: Add disparate impact calculation
docs: Update README with new examples
fix: Handle empty groups in metrics calculation

## Documentation

When adding new features, please:

- Update the README if needed
- Add docstrings to new functions/classes
- Add usage examples
- Update the CHANGELOG

## Questions?

If you have questions about contributing, feel free to:

- Open a discussion on GitHub
- Open an issue with the `question` label
- Reach out to the maintainers

## Recognition

Contributors will be recognized in our README. Thank you for making FairCheck better!

---

**Happy contributing!**
