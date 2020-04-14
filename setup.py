from setuptools import find_packages
from setuptools import setup

# Package meta-data.
NAME = "robupy"
DESCRIPTION = "An open-source package for robust optimization."
URL = "https://robupy.readthedocs.io/en/latest/"
EMAIL = ""
AUTHOR = "Maximilian Blesch & Philipp Eisenhauer"


setup(
    name=NAME,
    version="1.1",
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    license="MIT",
    include_package_data=True,
)
