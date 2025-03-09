#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

// Global mapping from call-stack (as string) to invocation count.
// Accesses to this map are protected by globalMutex.
static std::unordered_map<std::string, size_t> globalStackTraces;
static std::mutex globalMutex;

// A thread-local call-stack for each thread.
thread_local std::vector<std::string> tl_callStack;

// Helper function to join the elements of a call-stack vector using " -> " as a separator.
static std::string JoinCallStack(const std::vector<std::string>& cs) {
    std::string result;
    for (size_t i = 0; i < cs.size(); i++) {
        if (i > 0)
            result += " -> ";
        result += cs[i];
    }
    return result;
}

// The profiler hook function that will be called for various events.
static int profiler_hook(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
    // Determine what event this is.
    // (We only process CALL and RETURN events in this example.)
    switch (what) {
        case PyTrace_CALL: {
            // Get the function name from the current frame.
            const char *func_name = "<unknown>";
            PyObject *code = PyFrame_GetCode(frame);
            if (code) {
                PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
                if (name_obj) {
                    func_name = PyUnicode_AsUTF8(name_obj);
                }
                Py_XDECREF(name_obj);
            }
            // Record the function call by pushing it to the thread-local call stack.
            tl_callStack.push_back(std::string(func_name));
            // Build a complete string representation of the current call stack.
            std::string callStackStr = JoinCallStack(tl_callStack);
            // Update the global call-stack trace counter in a thread-safe way.
            {
                std::lock_guard<std::mutex> lock(globalMutex);
                globalStackTraces[callStackStr]++;
            }
            break;
        }
        case PyTrace_RETURN:
            // On function return, pop the call stack if not empty.
            if (!tl_callStack.empty()) {
                tl_callStack.pop_back();
            }
            break;
        // If needed, additional handling (e.g. for exceptions or line events) can be added here.
        default:
            break;
    }
    return 0;
}

// Python method to enable the profiler hook.
static PyObject *set_profile2(PyObject *self, PyObject *args) {
    // Use PyEval_SetTrace to set the hook. This causes profiler_hook to be called
    // on events (CALL, RETURN, etc.) for the current and future threads.
    PyEval_SetTrace(profiler_hook, NULL);
    Py_RETURN_NONE;
}

// Python method to disable the profiler hook.
static PyObject *unset_profile2(PyObject *self, PyObject *args) {
    PyEval_SetTrace(NULL, NULL);
    Py_RETURN_NONE;
}

// Python method to dump the recorded call stack traces along with their invocation counts.
// Returns a Python dictionary mapping call-stack strings to counts.
static PyObject *dump_callstacks(PyObject *self, PyObject *args) {
    PyObject *result = PyDict_New();
    if (!result)
        return NULL;
    // Lock the global map while iterating.
    {
        std::lock_guard<std::mutex> lock(globalMutex);
        for (const auto &pair : globalStackTraces) {
            PyObject *count = PyLong_FromSize_t(pair.second);
            PyDict_SetItemString(result, pair.first.c_str(), count);
            Py_DECREF(count);
        }
    }
    return result;
}

// Define the methods for this module.
static PyMethodDef Prof2Methods[] = {
    {"set_profile2", set_profile2, METH_NOARGS, "Enable C++-based threading profiler hook."},
    {"unset_profile2", unset_profile2, METH_NOARGS, "Disable C++-based threading profiler hook."},
    {"dump_callstacks", dump_callstacks, METH_NOARGS, "Dump recorded call stack traces and their invocation counts."},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Module definition.
static struct PyModuleDef cprofiler2_module = {
    PyModuleDef_HEAD_INIT,
    "cprofiler2",  // Module name
    "A C++-based profiler that tracks call stacks per thread and dumps call stack traces with counts.",
    -1,
    Prof2Methods
};

// Module initialization function.
PyMODINIT_FUNC PyInit_cprofiler2(void) {
    return PyModule_Create(&cprofiler2_module);
}
