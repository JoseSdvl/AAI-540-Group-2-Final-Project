from setuptools import setup, find_packages

setup(
    name="readmit",
    version="0.1.0",
    description="AAI-540 Group 2 - 30-day hospital readmission risk MLOps package",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.10",
)
