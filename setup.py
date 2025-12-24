"""Setup script for S3Rec with Low-rank AAP."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="s3rec_lowrank",
    version="1.0.0",
    author="Research Team",
    author_email="research@example.com",
    description="S3Rec with Low-rank Associated Attribute Prediction for Sequential Recommendation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/example/s3rec-lowrank",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "notebook": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "s3rec-preprocess=experiments.preprocess:main",
            "s3rec-pretrain=experiments.pretrain:main",
            "s3rec-finetune=experiments.finetune:main",
        ],
    },
)

