# Changelog

All notable changes to this project will be documented in this file.

The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to
[Semantic Versioning](https://semver.org/).

---

## [0.1.2] - 2026-01-13

### Changed
- Centralized version definition in a single source (`version.py`)
- Improved internal version consistency between runtime and packaging

---

## [0.1.1] - 2026-01-12

### Added
- Version compatibility tracking in `Torch2TransformerConfig`
- Runtime warning when loading checkpoints created with newer library versions

---

## [0.1.0] - 2026-01-11

### Added
- Initial public release
- Wrapper for arbitrary PyTorch models to integrate with Hugging Face `Trainer`
- Support for `save_pretrained()` and `from_pretrained()`
- Task-based wrapping (e.g. `causal_lm`)
