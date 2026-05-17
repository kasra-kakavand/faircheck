# Changelog

All notable changes to FairCheck will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Core `audit()` function for one-line fairness analysis
- `FairnessMetrics` class with comprehensive metric calculations
- `AuditResults` container with summary, report, and export methods
- Support for multiple sensitive attributes simultaneously
- Module-level convenience functions:
  - `demographic_parity()`
  - `equalized_odds()`
  - `equal_opportunity()`
  - `disparate_impact()`
- Comprehensive unit tests with pytest
- GitHub Actions CI/CD pipeline
- Multi-platform support (Linux, macOS, Windows)
- Multi-version Python support (3.8 - 3.12)

### Documentation
- Professional README with usage examples
- Inline docstrings for all public APIs
- Basic usage example in `examples/`

## [0.1.0] - 2026-05-17

### Added
- Initial project structure
- MIT License
- Core package skeleton
- Modern Python packaging with `pyproject.toml`

---

## Release Types

- **Major (X.0.0)** - Incompatible API changes
- **Minor (0.X.0)** - New features, backward compatible
- **Patch (0.0.X)** - Bug fixes, backward compatible

## Categories

- **Added** - New features
- **Changed** - Changes to existing functionality
- **Deprecated** - Soon-to-be removed features
- **Removed** - Removed features
- **Fixed** - Bug fixes
- **Security** - Security improvements
