from setuptools import setup, find_packages

setup(
    name="acres",
    version="0.1.0",
    author="Vaclav Volhejn",
    description="ML-based barcode sharpening.",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        # When running on GCP, tensorflow is guaranteed via `runtimeVersion: "1.5"`.
        # Specifying tensorflow version even seems to break GPU access.
        # "tensorflow == 1.5.0",
        "tensorflow",
        "tqdm == 4.23",
        "google-cloud-storage == 1.10.0",
    ]
)
