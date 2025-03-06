Build and install:

```
python setup.py sdist bdist_wheel
pip install .
```

Usage:

```
slim-trace /path/to/torch_profiler.json
```

```
parse-bt /path/to/backtrace.txt
```
