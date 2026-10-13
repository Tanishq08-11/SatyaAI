# Contributing to SATYA AI

Thank you for your interest in contributing to **SATYA AI**! We welcome contributions that help improve evidence verification, fact-checking accuracy, and misinformation mitigation.

---

## Code of Conduct
Please review and adhere to our [Code of Conduct](CODE_OF_CONDUCT.md) in all project interactions.

---

## Getting Started

### 1. Fork & Clone
1. Fork the repository on GitHub: `https://github.com/Tanishq08-11/SatyaAI`
2. Clone your fork locally:
   ```bash
   git clone https://github.com/<your-username>/SatyaAI.git
   cd SatyaAI
   ```

### 2. Environment Setup
Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Set up environment variables:
```bash
cp .env.example .env
```

---

## Development Workflow

1. Create a descriptive branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
2. Write clean, modular code that complies with the project's evidence-first principles:
   - **Never fabricate evidence or fact-checks.**
   - Preserve conservative sufficiency gate policies.
3. Run the test suite:
   ```bash
   pytest
   ```
4. Commit your changes following clear commit conventions:
   ```bash
   git commit -m "feat(verifier): add support for new evidence provider"
   ```
5. Push to your fork and submit a Pull Request against `main`.

---

## Reporting Issues
- Use [GitHub Issues](https://github.com/Tanishq08-11/SatyaAI/issues) to report bugs or request features.
- Please use the provided [Bug Report](.github/ISSUE_TEMPLATE/bug_report.md) or [Feature Request](.github/ISSUE_TEMPLATE/feature_request.md) templates.
