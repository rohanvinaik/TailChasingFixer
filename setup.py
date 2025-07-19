"""Setup configuration for tail-chasing-detector PyPI package."""

import os
from setuptools import setup, find_packages

# Read the README for PyPI
with open("README_PYPI.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from package
version_file = os.path.join(os.path.dirname(__file__), "tailchasing", "__init__.py")
with open(version_file) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        version = "0.1.0"

setup(
    name="tail-chasing-detector",
    version=version,
    author="Rohan Vinaik",
    author_email="rohanpvinaik@gmail.com",
    description="Detect LLM-assisted tail-chasing anti-patterns in Python code",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohanvinaik/TailChasingFixer",
    project_urls={
        "Bug Tracker": "https://github.com/rohanvinaik/TailChasingFixer/issues",
        "Documentation": "https://tail-chasing-detector.readthedocs.io",
        "Source Code": "https://github.com/rohanvinaik/TailChasingFixer",
    },
    packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=[
        "typing-extensions>=4.5",
        "pyyaml>=6.0",
        "click>=8.0",
        "rich>=13.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "coverage>=6.0",
            "ruff>=0.1",
            "mypy>=1.0",
            "black>=23.0",
            "isort>=5.0",
            "build>=0.10",
            "twine>=4.0",
        ],
        "visualization": [
            "plotly>=5.0",
            "matplotlib>=3.5",
            "seaborn>=0.12",
            "networkx>=3.0",
        ],
        "ml": [
            "scikit-learn>=1.3",
            "umap-learn>=0.5",
            "faiss-cpu>=1.7",
        ],
        "performance": [
            "numba>=0.57",
            "ray>=2.0",
            "dask>=2023.1",
        ],
        "cli": [
            "prompt-toolkit>=3.0",
            "watchdog>=3.0",
        ],
        "lsp": [
            "pygls>=1.0",
            "lsprotocol>=2023.0",
        ],
        "all": [
            "tail-chasing-detector[visualization,ml,performance,cli,lsp]",
        ],
    },
    entry_points={
        "console_scripts": [
            "tailchasing=tailchasing.cli:main",
            "tail-chasing=tailchasing.cli:main",  # Alternative name
        ],
    },
    include_package_data=True,
    package_data={
        "tailchasing": ["py.typed"],
    },
    keywords=[
        "code-quality",
        "static-analysis",
        "linting",
        "llm",
        "ai-assisted-development",
        "semantic-analysis",
        "hypervector",
        "tail-chasing",
    ],
)