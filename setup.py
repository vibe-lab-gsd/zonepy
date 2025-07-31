from setuptools import setup, find_packages

setup(
    name="zonepy",  
    version="0.5.0",  
    description="A Python package for zoning analysis",  
    long_description=open("README.md").read(),  
    long_description_content_type="text/markdown",  
    author="Houpu Li",  
    author_email="houpu_li@gsd.harvard.edu", 
    url="https://github.com/HOUPU1993/zonepy",  
    license="MIT",  
    packages=find_packages(),  
    include_package_data=True,  
    install_requires=[
        # Add your dependencies here, e.g.:
        # "numpy>=1.21.0",
        # "pandas>=1.3.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",  
        "License :: OSI Approved :: MIT License",  
        "Operating System :: OS Independent",  
    ],
    python_requires=">=3.7",  
)