import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imutils",
    version="0.0.2.0",
    author="Ulises Rey",
    author_email="Lukas.Hille@univie.ac.at",
    description="A small package with image processing utilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zimmer-lab/imutils",
    packages=(setuptools.find_packages()+['MicroscopeDataReader']),
    #package_dir={"":"},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
)