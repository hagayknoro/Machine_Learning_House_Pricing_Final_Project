from setuptools import setup, find_packages

setup(
    name="ml_project",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'jupyter'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A machine learning project",
    python_requires='>=3.7',
) 