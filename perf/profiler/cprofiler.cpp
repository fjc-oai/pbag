#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

// Global mapping from concatenated call-stack string to invocation count.
static std::unordered_map<std::string, size_t> stats;

// The profiler hook that will be called on various events (per Python thread).
static int profiler_hook(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
    // unsigned long threadId = PyThread_get_thread_ident();
    // Ensure we have a call-stack vector for this thread.
    // {
    //     if (pythonThreadCallStacks.find(threadId) == pythonThreadCallStacks.end()) {
            // pythonThreadCallStacks[threadId] = std::vector<std::string>();
        // }
    // }

    switch (what) {
        case PyTrace_CALL: {
            // Get the function name from the code object.
            // const char *filename = "<unknown>";
            const char *func_name = nullptr;
            int lineno = -1;
            PyObject *code = (PyObject *)PyFrame_GetCode(frame);
            if (code) {
                // PyObject *filename_obj = PyObject_GetAttrString(code, "co_filename");
                // if (!filename_obj) {
                //     return 0;
                // }
                // filename = PyUnicode_AsUTF8(filename_obj);
                // if (strstr(filename, substr) == NULL) {
                //     return 0;
                // }
                PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
                if (name_obj) {
                    func_name = PyUnicode_AsUTF8(name_obj);
                }
                PyObject *lineno_obj = PyObject_GetAttrString(code, "co_firstlineno");
                if (lineno_obj) {
                    lineno = PyLong_AsLong(lineno_obj);
                }
                Py_XDECREF(name_obj);
                std::string name = std::string(func_name) + ":" + std::to_string(lineno);
                stats[name]++;
            }
            break;
        }
        case PyTrace_RETURN: {
            // On function return, remove the last function from the call stack.
            // if (!pythonThreadCallStacks[threadId].empty()) {
                // pythonThreadCallStacks[threadId].pop_back();
            // }
            break;
        }
        default:
            break;
    }
    return 0;
}

// Python method to enable the profiler hook.
static PyObject *set_profiler(PyObject *self, PyObject *args) {
    // Set the trace function so that profiler_hook is called on events (CALL, RETURN, etc.)
    PyEval_SetTrace(profiler_hook, NULL);
    Py_RETURN_NONE;
}

// Python method to disable the profiler hook.
static PyObject *unset_profiler(PyObject *self, PyObject *args) {
    PyEval_SetTrace(NULL, NULL);
    Py_RETURN_NONE;
}

// Python method to dump the aggregated call-stack traces along with their invocation counts.
// Returns a Python dictionary mapping call-stack strings to counts.
static PyObject *dump_stats(PyObject *self, PyObject *args) {
    PyObject *result = PyDict_New();
    if (stats.empty()) {
        return result;
    }
    for (const auto &pair : stats) {
        PyObject *count = PyLong_FromSize_t(pair.second);
        PyObject *key = PyUnicode_FromString(pair.first.c_str());
        PyDict_SetItem(result, key, count);
    }
    return result;
}

// Define the methods for this module.
static PyMethodDef ProfMethods[] = {
    {"set_profiler", set_profiler, METH_NOARGS, "Enable the profiler hook for each Python thread."},
    {"unset_profiler", unset_profiler, METH_NOARGS, "Disable the profiler hook."},
    {"dump_stats", dump_stats, METH_NOARGS, "Dump aggregated call stack traces and their invocation counts."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition.
static struct PyModuleDef cprofiler_module = {
    PyModuleDef_HEAD_INIT,
    "cprofiler",  // Module name
    "A C++-based profiler that tracks call stacks for each Python thread and dumps aggregated call-stack traces with counts.",
    -1,
    ProfMethods
};

// Module initialization function.
PyMODINIT_FUNC PyInit_cprofiler(void) {
    return PyModule_Create(&cprofiler_module);
} 