from setuptools import setup, find_packages
import os

# Read the version from VERSION_NUMBER file
with open('VERSION_NUMBER', 'r') as f:
    version = f.read().strip()

# Read the requirements from requirements.txt
with open('requirements.txt', 'r') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Read the long description from README.md
long_description = ""
if os.path.exists('README.md'):
    with open('README.md', 'r', encoding='utf-8') as f:
        long_description = f.read()

setup(
    name="AITools",
    version=version,
    description="A Python package for AI model deployment and inference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ChisenV",
    author_email="",  # Add email if available
    url="https://github.com/ChisenV/AITools",
    packages=find_packages(include=['AITools', 'AITools.*']),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
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
    include_package_data=True,
    zip_safe=False,
    license="MIT",
    keywords="ai, machine learning, model deployment, inference, onnx, tensorrt",
)
