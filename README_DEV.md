# Releasing new version

To release a new version into the world make sure the following points are followed:

1. Update the package version, both in [pyproject.toml](pyproject.toml) and in the package itself [congrads/__init__.py](congrads/__init__.py)
2. Regenerate the distribution archives with ```python3 -m build```
3. Publish the code to PyPi with ```python3 -m twine upload --repository pypi dist/*```

For more information, refer to [this guide](https://packaging.python.org/en/latest/tutorials/packaging-projects/).