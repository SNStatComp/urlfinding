import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="urlfinding",
    version="0.0.3",
    author="Dick Windmeijer",
    author_email="dwnr@cbs.nl",
    description="Generic software for finding websites of enterprises using a search engine and machine learning.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/SNStatComp/urlfinding.git",
    packages=setuptools.find_packages(),
	install_requires=[
		"google-api-python-client",
        "flatten_json",
        "jellyfish",
        "pycm",
        "yellowbrick",
        "selenium",
        "selenium-wire",
        "pyderman"
	],
    classifiers=[
        "Development Status :: 4 - Beta"
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    zip_safe=False
)