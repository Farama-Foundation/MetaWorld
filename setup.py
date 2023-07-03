from setuptools import find_packages, setup


# Required dependencies
required = [
    # Please keep alphabetized
    "gymnasium>=0.28.1",
    "mujoco",
    "numpy>=1.18",
    "scipy",
]


# Development dependencies
extras = dict()
extras["dev"] = [
    # Please keep alphabetized
    "ipdb",
    "memory_profiler",
    "pylint",
    "pyquaternion==0.9.5",
    "pytest>=4.4.0",  # Required for pytest-xdist
    "pytest-xdist",
    "scipy",
]


setup(
    name="metaworld",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=required,
    extras_require=extras,
    python_requires='>3.7'
)
