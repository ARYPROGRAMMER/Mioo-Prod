from setuptools import setup, find_packages

setup(
    name="rl_model_mio",
    version="0.1.0",
    description="Reinforcement Learning for Adaptive Education Content",
    author="AI Tutor Team",
    author_email="ai_tutor@example.com",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
        "tqdm>=4.46.0",
        "matplotlib>=3.2.1",
        "pydantic>=1.8.0",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
)
