# Copyright (c) 2022 Graphcore Ltd. All rights reserved.

import io
import os
from pathlib import Path
from glob import glob

from setuptools import setup


def read(*paths, **kwargs):
    """Read the contents of a text file safely.
    >>> read("project_name", "VERSION")
    '0.1.0'
    >>> read("README.md")
    ...
    """

    content = ""
    with io.open(
        os.path.join(os.path.dirname(__file__), *paths),
        encoding=kwargs.get("encoding", "utf8"),
    ) as open_file:
        content = open_file.read().strip()
    return content


def read_requirements(path):
    return [line.strip() for line in read(path).split("\n") if not line.startswith(('"', "#", "-"))]


setup(
    name="popxl-addons",
    description="popxl.addons",
    long_description="file: README.md",
    long_description_content_type="text/markdown",
    license="MIT License",
    author="Graphcore Ltd.",
    url="https://github.com/graphcore/popxl-addons",
    # download_urls = "https://pypi.org/project/popxl-addons",
    project_urls={
        "Code": "https://github.com/graphcore/popxl-addons",
        # "Issue tracker": "https://github.com/graphcore/popxl-addons/issues",
    },
    classifiers=[  # Optional
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
    install_requires=read_requirements("requirements.txt"),
    extras_require={"dev": read_requirements("requirements-dev.txt")},
    python_requires=">=3.8",
    packages=["popxl_addons"],
    package_data={
        "popxl_addons":
        # Paths need to be relative to `popxl_addons/` folder
        [
            os.path.join(*Path(f).parts[1:])
            for ext in [".py", ".cpp", ".hpp"]
            for f in glob(f"popxl_addons/**/*{ext}", recursive=True)
        ]
    },
)
