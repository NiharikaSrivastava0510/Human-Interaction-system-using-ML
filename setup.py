from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="human-activity-recognition",
    version="1.0.0",
    author="Niharika Srivastava",
    author_email="niharika051095@gmail.com",
    description="Machine learning pipeline for human activity recognition from accelerometer data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NiharikaSrivastava0510/human-activity-recognition",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.10",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
        "tensorflow>=2.13.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyter>=1.0.0",
    ],
)
