from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="inverse_correlation_nb",
    version="1.0.0",
    author="Antigravity",
    description="Inverse-Correlation Naive Bayes (IC-NB) - A mathematically principled, dependency-aware extension to Gaussian Naive Bayes.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/inverse_correlation_nb",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires='>=3.7',
    install_requires=[
        'numpy',
        'scipy',
        'pandas',
        'scikit-learn'
    ],
)
