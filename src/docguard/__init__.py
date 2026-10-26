from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("docguard-rag")
except PackageNotFoundError:
    __version__ = "0.0.0+uninstalled"

__all__ = ["__version__"]
