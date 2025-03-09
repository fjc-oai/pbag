#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <iostream>
#include <string>
#include <vector>

// Global call-stack variable that stores function names.
static std::vector<std::string> callStack;

// A simple profiling function using accessor functions.
static int
profilefunc(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg)
{
    // return 0;
    const char *event;
    switch (what)
    {
    case PyTrace_CALL:
        event = "call";
        break;
    case PyTrace_RETURN:
        event = "return";
        break;
    case PyTrace_EXCEPTION:
        event = "exception";
        break;
    case PyTrace_LINE:
        event = "line";
        break;
    default:
        event = "other";
        break;
    }

    // Get the code object and extract the function name.
    PyObject *code = (PyObject *)PyFrame_GetCode(frame);
    const char *func_name = "<unknown>";
    if (code)
    {
        PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
        if (name_obj)
        {
            func_name = PyUnicode_AsUTF8(name_obj);
        }
        Py_XDECREF(name_obj);
    }
    return 0;

    std::string output = "Event: " + std::string(event) + ", Function: " + std::string(func_name);
    std::cout << output << std::endl;

    // Handle events:
    if (what == PyTrace_CALL)
    {
        // When a new function is called, push its name to the global vector.
        callStack.push_back(func_name);
    }
    else if (what == PyTrace_RETURN)
    {
        // When a function returns, pop its name and print it along with its parent functions.
        if (!callStack.empty())
        {
            std::string current_func = callStack.back();
            callStack.pop_back();
            std::cout << "Returning from function: " << current_func;
            if (!callStack.empty())
            {
                std::cout << ", Parent functions:";
                for (const auto &parent : callStack)
                {
                    std::cout << " " << parent;
                }
            }
            std::cout << std::endl;
        }
    }
    return 0;
}

// Python method to enable our profiler
static PyObject *
set_profile(PyObject *self, PyObject *args)
{
    // PyEval_SetProfile(profilefunc, NULL);
    PyEval_SetTrace(profilefunc, NULL);
    Py_RETURN_NONE;
}

// Python method to disable profiling
static PyObject *
unset_profile(PyObject *self, PyObject *args)
{
    // PyEval_SetProfile(NULL, NULL);
    PyEval_SetProfile(NULL, NULL);
    Py_RETURN_NONE;
}

// Method definitions for our module
static PyMethodDef ProfMethods[] = {
    {"set_profile", set_profile, METH_NOARGS, "Enable C++-based profiling."},
    {"unset_profile", unset_profile, METH_NOARGS, "Disable C++-based profiling."},
    {NULL, NULL, 0, NULL}};

// Module definition
static struct PyModuleDef cprofiler_module = {
    PyModuleDef_HEAD_INIT,
    "cprofiler",                                           // Module name
    "A simple C++-based profiler using the Python C API.", // Module docstring
    -1,
    ProfMethods};

// Module initialization function
PyMODINIT_FUNC
PyInit_cprofiler(void)
{
    return PyModule_Create(&cprofiler_module);
}

// } // extern "C"