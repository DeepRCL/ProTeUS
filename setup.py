from setuptools import setup, find_packages
import os

# Read requirements from requirements.txt
def read_requirements():
    """Read and parse requirements from requirements.txt file.
    
    Returns:
        List of package requirements, excluding comments and empty lines.
    """
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return requirements

# Read README for long description
def read_readme():
    """Read README.md file for package description.
    
    Returns:
        Content of README.md file if it exists, otherwise a default description.
    """
    if os.path.exists("README.md"):
        with open("README.md", "r", encoding="utf-8") as f:
            return f.read()
    return "ProTeUS: A Spatio-Temporal Enhanced Ultrasound-Based Framework for Prostate Cancer Detection"

setup(
    name="proteus",
    version="1.0.0",
    description="ProTeUS: A Spatio-Temporal Enhanced Ultrasound-Based Framework for Prostate Cancer Detection",
    long_description=read_readme(),
    author="Tarek Elghareb et al.",
    url="https://github.com/DeepRCL/ProTeUS",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=read_requirements(),
)
