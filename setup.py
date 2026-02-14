from setuptools import find_packages, setup


setup(
    name="rnode",
    version="0.1.0",
    description="Random Batch Neural ODEs companion library",
    packages=find_packages(include=["rnode", "rnode.*"]),
    python_requires=">=3.10",
)
