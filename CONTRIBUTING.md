Thank you for your interest in contributing to this project! Below are a few quick guidelines to help you get started.

1. Fork the repository and create your branch from `master`.
2. Make changes on a feature branch and include tests where applicable.
3. Keep changes focused and open a pull request describing what you changed and why.

Local development
-----------------

- Create and activate a virtual environment:

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

- Install development dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install flake8
```

- Run the small smoke-check (this does not run the heavy model by default):

```bash
# The included helper prints a summary into tools/response.json when run locally
python tools/run_inference_sample.py || true
```

When opening a PR
-----------------

- Describe the intent, and reference any related issues.
- Keep changes scoped to one logical idea per PR.
- A maintainer will review and request changes as needed.

Security
--------

If you discover a security vulnerability, please open a private issue or reach out to the maintainers instead of posting details publicly.
