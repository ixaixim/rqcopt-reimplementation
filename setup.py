from setuptools import setup

setup(
    name="rqcopt_mpo",
    version="1.0.0",
    author="Neel Misciasci",
    author_email="neel.miscia@gmail.com",
    packages=["rqcopt_mpo"],
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "psutil",
        "PyYAML",
        "jax",
        "jaxlib",
    ],
)