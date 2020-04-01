from setuptools import find_packages
from setuptools import setup

# Package meta-data.
NAME = "robupy"
DESCRIPTION = (
    ""
)
URL = ""
EMAIL = ""
AUTHOR = ""


setup(
    name=NAME,
    version="0.1.0",
    description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=find_packages(exclude=("tests",)),
    license="MIT",
    include_package_data=True,
)
