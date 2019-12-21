from setuptools import setup


PROJECT_ROOT = dirname(realpath(__file__))
REQUIREMENTS_FILE = join(PROJECT_ROOT, "requirements.txt")


with open(REQUIREMENTS_FILE) as f:
    install_reqs = f.read().splitlines()


setup(
    name="covadapt",
    version="0.1",
    url="http://github.com/aseyboldt/covadapt",
    author="Adrian Seyboldt",
    author_email="adrian.seyboldt@gmail.com",
    license="MIT",
    packages=["covadapt"],
    install_requires=install_reqs,
    zip_safe=False,
)
