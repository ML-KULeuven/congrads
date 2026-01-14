try:  # noqa: D104
    from importlib.metadata import version as get_version  # Python 3.8+
except ImportError:
    from pkg_resources import (
        get_distribution as get_version,
    )  # Fallback for older versions

try:
    version = get_version("congrads")  # Replace with your package name
except Exception:
    version = "0.0.0"  # Fallback if the package isn't installed
