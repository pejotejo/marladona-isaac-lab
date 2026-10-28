from setuptools import find_packages, setup

setup(
    name="rsl_marl",
    version="1.0.0",
    packages=find_packages(),
    license="BSD-3",
    description="Fast and simple RL algorithms implemented in pytorch. Adapted for MARLadona.",
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.7",
        "numpy>=1.24",
        "GitPython",
        "tensorboard",
        "rsl-rl-lib>=5.4.2,<6",
    ],
)
