# Contributing to phantom-detect

Contributions are welcome. Here's how to get started.

## Setup

```bash
git clone https://github.com/ScrappinR/phantom-detect.git
cd phantom-detect
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

This project uses [ruff](https://docs.astral.sh/ruff/) for linting:

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Submitting Changes

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Make your changes
4. Run tests (`pytest`)
5. Commit with a descriptive message
6. Push to your fork
7. Open a pull request

## Reporting Issues

Open an issue on GitHub with:
- What you expected
- What happened
- Steps to reproduce
- Python version and OS

## Adding Detection Methods

If you've identified a new covert channel type in LLM outputs:

1. Add the channel type to `ChannelType` in `src/phantom_detect/types.py`
2. Implement detection logic (see `statistical.py` for the pattern)
3. Add tests
4. Document the channel in the README

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.
