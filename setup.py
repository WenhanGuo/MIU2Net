from setuptools import setup, find_packages

setup(
    name="miu2net",
    version="1.0.0",
    author="Wenhan Guo",
    author_email="wgaa2021@mymail.pomona.edu",
    description="Weak lensing mapping of dark matter based on ML methods",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/WenhanGuo/MIU2Net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=[
        "numpy",
        "astropy",
    ],
    license="MIT",
)