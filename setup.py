from setuptools import setup, find_packages

setup(
    name="miu2net",
    version="0.1.0",
    author="Wenhan Guo",
    author_email="gwh1703@gmail.com",
    description="Weak lensing mapping of dark matter based on ML methods",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/WenhanGuo/MIU2Net",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    python_requires='>=3.7',
    install_requires=[
        "numpy",
        "scipy",
        "astropy>=4.0",
        "scikit-learn>=0.20",
    ],
    license="MIT",
)