#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

// Global mapping from concatenated call-stack string to invocation count.
static std::unordered_map<std::string, size_t> globalStackTraces;

// Global mapping from Python thread ID to its current call stack.
static std::unordered_map<unsigned long, std::vector<std::string>> pythonThreadCallStacks;
static std::mutex globalMutex;

// Helper function to join elements of a call-stack vector using " -> " as a separator.
static std::string JoinCallStack(const std::vector<std::string>& cs) {
    std::string result;
    for (size_t i = 0; i < cs.size(); ++i) {
        if (i > 0) {
            result += " -> ";
        }
        result += cs[i];
    }
    return result;
}


// The profiler hook that will be called on various events (per Python thread).
static int profiler_hook(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
    // Get the current Python thread id.
    unsigned long threadId = PyThread_get_thread_ident();

    // Ensure we have a call-stack vector for this thread.
    {
        if (pythonThreadCallStacks.find(threadId) == pythonThreadCallStacks.end()) {
            pythonThreadCallStacks[threadId] = std::vector<std::string>();
        }
    }

    switch (what) {
        case PyTrace_CALL: {
            // Get the function name from the code object.
            const char *func_name = "<unknown>";
            const char *filename = "<unknown>";
            const char *substr = "perf";
            int lineno = -1;
            PyObject *code = (PyObject *)PyFrame_GetCode(frame);
            if (code) {
                PyObject *filename_obj = PyObject_GetAttrString(code, "co_filename");
                if (!filename_obj) {
                    return 0;
                }
                filename = PyUnicode_AsUTF8(filename_obj);
                if (strstr(filename, substr) == NULL) {
                    return 0;
                }

                PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
                if (name_obj) {
                    func_name = PyUnicode_AsUTF8(name_obj);
                }
                PyObject *lineno_obj = PyObject_GetAttrString(code, "co_firstlineno");
                if (lineno_obj) {
                    lineno = PyLong_AsLong(lineno_obj);
                }
                Py_XDECREF(name_obj);

                std::string name = std::string(filename) + " " + std::to_string(lineno) + " " + std::string(func_name);
                std::cout << "name = " << name << std::endl;
                pythonThreadCallStacks[threadId].push_back(name);
            }
            break;
        }
        case PyTrace_RETURN: {
            // On function return, remove the last function from the call stack.
            pythonThreadCallStacks[threadId].push_back("RET");
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
static PyObject *dump_callstacks(PyObject *self, PyObject *args) {
    PyObject *result = PyDict_New();
    if (!result)
        return NULL;
    // for (const auto &pair : globalStackTraces) {
    //     PyObject list = Py
    //     PyObject *count = PyLong_FromSize_t(pair.second);
    //     PyDict_SetItemString(result, pair.first.c_str(), count);
    //     Py_DECREF(count);
    // }
    PyObject *res = PyList_New(0);
    std::cout << "pythonThreadCallStacks.size() = " << pythonThreadCallStacks.size() << std::endl;
    for (const auto &pair : pythonThreadCallStacks) {
        std::cout << "pair.first = " << pair.first << ", pair.second.size() = " << pair.second.size() << std::endl;
        PyObject *list = PyList_New(0);
        for (const auto &callStack : pair.second) {
            // std::cout << "callStack = " << callStack << std::endl;
            PyObject *item = PyUnicode_FromString(callStack.c_str());
            PyList_Append(list, item);
        }
        PyList_Append(res, list);
    }
    return res;
}

// Define the methods for this module.
static PyMethodDef ProfMethods[] = {
    {"set_profiler", set_profiler, METH_NOARGS, "Enable the profiler hook for each Python thread."},
    {"unset_profiler", unset_profiler, METH_NOARGS, "Disable the profiler hook."},
    {"dump_callstacks", dump_callstacks, METH_NOARGS, "Dump aggregated call stack traces and their invocation counts."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition.
static struct PyModuleDef cpython_thread_profiler_module = {
    PyModuleDef_HEAD_INIT,
    "cpython_thread_profiler",  // Module name
    "A C++-based profiler that tracks call stacks for each Python thread and dumps aggregated call-stack traces with counts.",
    -1,
    ProfMethods
};

// Module initialization function.
PyMODINIT_FUNC PyInit_cpython_thread_profiler(void) {
    return PyModule_Create(&cpython_thread_profiler_module);
} 