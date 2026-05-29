# Contributing to SkillForge

Thank you for your interest in contributing to SkillForge!

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/skillforge.git
   cd skillforge
   ```
3. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
4. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

## Development Guidelines

- **Python version**: 3.10+ required.
- **Standard library only**: Core modules use no external dependencies. Optional extras (`viz`, `dev`) are for tooling only.
- **Type hints**: Use type annotations on all public functions and classes.
- **Docstrings**: Follow Google-style docstrings for all public APIs.
- **Tests**: Write tests for all new functionality. Place tests in the `tests/` directory mirroring the source structure.

## Running Tests

```bash
pytest
```

With coverage:

```bash
pytest --cov=skillforge
```

## Submitting Changes

1. Ensure all tests pass: `pytest`
2. Commit with a clear, descriptive message.
3. Push to your fork and open a Pull Request against `main`.
4. Describe what your changes do and why.

## Code Style

- Follow PEP 8 conventions.
- Keep functions focused and small.
- Prefer explicit over implicit behavior.

## Reporting Issues

Open an issue on GitHub with:
- A clear description of the problem or feature request
- Steps to reproduce (for bugs)
- Your Python version and OS

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
