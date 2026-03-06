---
icon: lucide/heart-handshake
---

# Contributing

Thank you for your interest in contributing to fAIr Models! This project is
part of [HOT — Humanitarian OpenStreetMap Team](https://www.hotosm.org/) and
powers AI-assisted mapping for humanitarian response.

## Ways to Contribute

=== ":lucide-box: Contribute a Model"

    The primary contribution path is adding a new base model to fAIr. See the
    [Contributing a Model](model.md) guide for the full specification.

=== ":lucide-wrench: Improve the Library"

    Bug fixes, features, and documentation improvements to the `fair-py-ops`
    package are welcome.

=== ":lucide-circle-alert: Report Issues"

    Open an issue on [GitHub](https://github.com/hotosm/fAIr-models/issues) with:

    - Clear description of the problem or suggestion
    - Steps to reproduce (for bugs)
    - Expected vs actual behavior

## Development Setup

```bash title="Set up local dev environment"
git clone https://github.com/hotosm/fAIr-models.git
cd fAIr-models
make setup
```

## Code Standards

!!! tip

    - **Linting, formatting & type checking:** `ruff` + `ty` : run `make lint` before submitting
    - **Tests:** `pytest` : run `make test`, all tests must pass
    - **Validation:** run `make validate` to check STAC items and model pipelines
    - **Commits:** [Conventional Commits](https://www.conventionalcommits.org/)
      enforced via `commitizen` and pre-commit hooks

## Pull Request Process

1. Fork the repository and create a feature branch
2. Make your changes with clear, atomic commits
3. Ensure all checks pass:

    ```bash title="Run all checks"
    make lint
    make test
    make validate
    ```

4. Open a PR against `master` with a clear description
5. CI will run lint, typecheck, tests, STAC validation, and model validation
6. Maintainers will review and provide feedback

## License

!!! warning

    By contributing, you agree that your contributions will be licensed under the
    [AGPL-3.0](https://www.gnu.org/licenses/agpl-3.0.html) license.
