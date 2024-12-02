Example package following python packaging tutorail https://packaging.python.org/en/latest/tutorials/packaging-projects/

- Build the package:
  
    `python -m build`

- Upload the package (don't forget to update the version number and clear up the dist folder first)

    `python -m twine upload --repository testpypi dist/*`

- Install the package:

    `python -m pip install --index-url https://test.pypi.org/simple --no-deps monitors`

- Upgrade the package:

    `python -m pip install --upgrade --index-url https://test.pypi.org/simple --no-deps monitors`

- Use the package

    ```
    python
    >>> from pd3220u.spec import print_spec
    >>> print_spec()
    ```
- Uninstall the package

  `python -m pip uninstall monitors`
