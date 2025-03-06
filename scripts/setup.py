from setuptools import find_packages, setup

setup(
    name="scripts",
    version="0.1.0",
    packages=find_packages(),
    description="A collection of frequently used Python scripts",
    author="fjc",
    author_email="fmarsf@gmail.com",
    install_requires=[
        # list any dependencies your package needs, e.g.,
        # "requests>=2.25.1",
    ],
    entry_points={
        "console_scripts": [
            "slim-trace=scripts.slim_trace:main",
            "parse-bt=scripts.parse_backtrace:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # choose your license
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)