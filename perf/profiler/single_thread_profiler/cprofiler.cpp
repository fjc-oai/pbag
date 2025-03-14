#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

static std::vector<std::int64_t> start_times;
static std::vector<std::string> function_names;
static std::unordered_map<std::string, std::int64_t> stats;

static int profiler_hook(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
    switch (what) {
        case PyTrace_CALL: {
            const char *func_name = nullptr;
            int lineno = -1;
            PyObject *code = (PyObject *)PyFrame_GetCode(frame);
            if (code) {
                PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
                if (name_obj) {
                    func_name = PyUnicode_AsUTF8(name_obj);
                    // Py_XDECREF(name_obj);
                }
                PyObject *lineno_obj = PyObject_GetAttrString(code, "co_firstlineno");
                if (lineno_obj) {
                    lineno = PyLong_AsLong(lineno_obj);
                    // Py_XDECREF(lineno_obj);
                }
                std::string name = std::string(func_name) + ":" + std::to_string(lineno);
                start_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count());
                function_names.push_back(name);
            }
            break;
        }
        case PyTrace_RETURN: {
            auto end_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count();
            if (start_times.empty()) {
                break;
            }
            auto start_time = start_times.back();
            start_times.pop_back();
            if (function_names.empty()) {
                break;
            }
            auto name = function_names.back();
            function_names.pop_back();
            std::int64_t duration = end_time - start_time;
            std::string stacktrace = "";
            for (size_t i = 0; i < function_names.size(); ++i) {
                stacktrace += function_names[i] + ";";
            }
            stacktrace += name;
            if (stats.find(stacktrace) == stats.end()) {
                stats[stacktrace] = 0;
            }
            stats[stacktrace] += duration;
        }
        break;
    }
    return 0;
}

static PyObject *set_profiler(PyObject *self, PyObject *args) {
    PyEval_SetTrace(profiler_hook, NULL);
    Py_RETURN_NONE;
}

static PyObject *unset_profiler(PyObject *self, PyObject *args) {
    PyEval_SetTrace(NULL, NULL);
    Py_RETURN_NONE;
}

static PyObject *dump_stats(PyObject *self, PyObject *args) {
    PyObject *result = PyDict_New();
    if (stats.empty()) {
        return result;
    }
    for (const auto &pair : stats) {
        PyObject *count = PyLong_FromLong(pair.second);
        PyObject *key = PyUnicode_FromString(pair.first.c_str());
        PyDict_SetItem(result, key, count);
        // Py_XDECREF(key);
        // Py_XDECREF(count);
    }
    return result;
}

static PyMethodDef ProfMethods[] = {
    {"set_profiler", set_profiler, METH_NOARGS, "Enable the profiler hook for each Python thread."},
    {"unset_profiler", unset_profiler, METH_NOARGS, "Disable the profiler hook."},
    {"dump_stats", dump_stats, METH_NOARGS, "Dump aggregated call stack traces and their invocation counts."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cprofiler_module = {
    PyModuleDef_HEAD_INIT,
    "cprofiler",
    "A C++-based profiler that tracks call stacks for each Python thread and dumps aggregated call-stack traces with counts.",
    -1,
    ProfMethods
};

PyMODINIT_FUNC PyInit_cprofiler(void) {
    return PyModule_Create(&cprofiler_module);
}