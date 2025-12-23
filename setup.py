from setuptools import setup, find_packages

setup(
    name="player-reid",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "ultralytics",
        "torch",
        "torchreid",
        "opencv-python",
        "pytesseract",
        "numpy",
        "scikit-learn",
        "pyyaml",
        "matplotlib",
    ],
)
