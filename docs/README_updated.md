# exajoule
energy modelling agents codename exajoule
...

Installation Requirements

Minimum system requirements
- Operating system: Linux, macOS, or Windows (WSL recommended for Windows).
- Python: 3.10 or later.
- Disk space: minimum 1 GB free for the project and dependencies; more required if you download large datasets or model checkpoints.
- Memory: minimum 8 GB RAM; 16 GB+ recommended for comfortable use.
- Network: internet access to download packages, datasets, and model artifacts.
- User privileges: ability to install system packages and Python packages (or use a virtual environment).

Recommended system packages and tools
- git
- pip (latest)
- python3-dev / build-essential (or Xcode command line tools on macOS) — required for compiling any native extensions.
- virtualenv or venv (recommended to isolate project dependencies)

Optional (for GPU acceleration)
- NVIDIA GPU with driver installed.
- CUDA toolkit and cuDNN versions compatible with your chosen ML libraries (check the framework docs for exact versions).
- Install GPU-enabled binaries of libraries (for example, PyTorch with CUDA support).

Python package requirements
- Install Python dependencies from the project's requirements file (e.g., requirements.txt) or from pyproject/poetry if used.
- Example common packages: numpy, pandas, scipy, scikit-learn, torch or tensorflow (if used), and any project-specific packages listed in requirements.

Quick installation steps (example)
1. Clone the repository:
   git clone <repo-url>
   cd exajoule
2. Create and activate a virtual environment:
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate       # Windows
3. Install dependencies:
   pip install --upgrade pip
   pip install -r requirements.txt
4. If using GPU-enabled libraries, follow the framework's installation instructions to install the correct CUDA-enabled wheel.

Troubleshooting tips
- If a build step fails, ensure system build tools and python dev headers are installed.
- For dependency conflicts, create a fresh virtual environment.
- If you need specific versions of CUDA or drivers, consult the deep-learning framework's compatibility matrix.
- Check the project's README or docs for any project-specific setup steps (API keys, environment variables, data downloads).

Additional content to add
- Additional content to add