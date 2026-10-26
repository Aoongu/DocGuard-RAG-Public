from importlib.metadata import version

import docguard


def test_package_exposes_version() -> None:
    assert docguard.__version__ == version("docguard-rag")
