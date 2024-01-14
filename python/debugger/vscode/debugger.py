# https://github.com/microsoft/debugpy?tab=readme-ov-file
# https://code.visualstudio.com/docs/python/debugging


"""
1. python vscode_debugging.py
2. click, "Run and Debug"

launch.json
{
    // Use IntelliSense to learn about possible attributes.
    // Hover to view descriptions of existing attributes.
    // For more information, visit: https://go.microsoft.com/fwlink/?linkid=830387
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Python: Remote Attach",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost",
                "port": 1234
            },
            "pathMappings": [
                {
                    "localRoot": "${workspaceFolder}",
                    "remoteRoot": "/Users/fjc/code/pbag/"
                }
            ],
            "justMyCode": true
        }
    ]
}

"""
import debugpy
import os


def fn1(x):
    x += 1
    x = fn2(x)
    x += 1
    return x


def fn2(x):
    x += 2
    breakpoint()
    x = fn3(x)
    x += 2
    return x


def fn3(x):
    x += 3
    return x


def set_trace():
    debugpy.listen(("localhost", 1234))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
    debugpy.breakpoint()


def main():
    os.environ["PYTHONBREAKPOINT"] = "debugger.set_trace"
    x = 10
    fn1(x)


if __name__ == "__main__":
    main()
