from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
    'google-cloud-storage==1.6',
    'pandas>=0.22.0',
    'tensorflow>=1.8.0',
]

setup(
    name='package',
    author='',
    author_email='',
    version='0.1',
    description='',
    packages=find_packages(),
    include_package_data=True,
    install_requires=REQUIRED_PACKAGES,
)
