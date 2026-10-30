from setuptools import find_namespace_packages, setup

setup(
    name="rsl_marl",
    version="1.0.0",
    packages=find_namespace_packages(include=["rsl_marl*"], exclude=["*.__pycache__"]),
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch. Adapted for MARLadona.",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.10.0",
        "torchvision>=0.5.0",
        "numpy>=1.16.4",
        "GitPython",
    ],
)
