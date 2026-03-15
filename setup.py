"""
Setup configuration for the Satellite Urban Growth Analysis package.
"""

from setuptools import setup, find_packages
from pathlib import Path

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="satellite-urban-growth-analysis",
    version="1.0.0",
    description="Deep learning pipeline for satellite-based urban growth analysis of Vijayawada",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Urban Growth Analysis Team",
    python_requires=">=3.9",
    packages=find_packages(exclude=["tests*", "notebooks*", "scripts*"]),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "rasterio>=1.3.0",
        "geopandas>=0.14.0",
        "folium>=0.15.0",
        "earthengine-api>=0.1.370",
        "albumentations>=1.3.0",
        "scikit-learn>=1.3.0",
        "scipy>=1.11.0",
        "matplotlib>=3.7.0",
        "pyyaml>=6.0.0",
        "python-dotenv>=1.0.0",
        "tqdm>=4.65.0",
        "Pillow>=10.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "gee": [
            "earthengine-api>=0.1.370",
            "geemap>=0.27.0",
        ],
        "notebooks": [
            "jupyter>=1.0.0",
            "jupyterlab>=4.0.0",
            "ipywidgets>=8.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "download-data=scripts.download_data:main",
            "preprocess-data=scripts.preprocess_data:main",
            "train-model=scripts.train_model:main",
            "evaluate-model=scripts.evaluate_model:main",
            "generate-maps=scripts.generate_maps:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
)
