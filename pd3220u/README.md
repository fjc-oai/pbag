Example package following python packaging tutorail https://packaging.python.org/en/latest/tutorials/packaging-projects/

- Build the package:
  
    `python3 -m build`

- Upload the package (don't forget to update the version number and clear up the dist folder first)

    `python3 -m twine upload --repository testpypi dist/*`

- Install the package:

    `python3 -m pip install --index-url https://test.pypi.org/simple --no-deps pd3220u`

- Upgrade the package:

    `python3 -m pip install --upgrade --index-url https://test.pypi.org/simple --no-deps pd3220u`

- Use the package

    ```
    python3
    >>> from pd3220u import thunderbolt"
    >>> thunderbolt.spec()
    ```
- Uninstall the package

  `python3 -m pip uninstall pd3220u`
