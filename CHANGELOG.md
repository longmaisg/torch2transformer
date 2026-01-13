# Changelog

All notable changes to this project will be documented in this file.

The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
and adheres to
[Semantic Versioning](https://semver.org/).

---

## [0.2.0] - 2026-01-13

### Added
- Full Hugging Face `.generate()` support for wrapped PyTorch models
- `forward()` now returns `CausalLMOutput` for compatibility with generation and Trainer
- `prepare_inputs_for_generation()` implemented for autoregressive decoding
- Batch-safe generation examples in README

### Changed
- Removed redundant `TorchAdapter` class; contract is enforced at runtime
- HF wrapper now normalizes user model output to ensure `logits` and optional `loss`
- Updated `Torch2TransformerConfig` to include `is_decoder=True` and `use_cache=False` for generation

---

## [0.1.2] - 2026-01-12

### Changed
- Centralized version definition in `version.py`
- Improved runtime version consistency between Python code and packaging

---

## [0.1.1] - 2026-01-11

### Added
- Version compatibility checking in `Torch2TransformerConfig`
- Runtime warning when loading checkpoints from a newer library version

---

## [0.1.0] - 2026-01-10

### Added
- Initial public release
- Wrapper for arbitrary PyTorch models to integrate with Hugging Face `Trainer`
- Support for `save_pretrained()` and `from_pretrained()`
- Task-based wrapping (e.g. `causal_lm`)
