#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <string_view>



static std::unordered_map<std::string_view, size_t> file_name_2_pos;
static std::vector<std::string> pos_2_file_name;

static size_t get_pos(const char* name, std::unordered_map<std::string_view, size_t>& map, std::vector<std::string>& vec) {
    auto it = map.find(name);
    if (it != map.end())
        return it->second;
    size_t pos = vec.size();
    vec.push_back(std::string(name));
    map[vec.back()] = pos;
    return pos;
}

size_t get_file_name_pos(const char *file_name) {
    return get_pos(file_name, file_name_2_pos, pos_2_file_name);
}

static std::unordered_map<std::string_view, size_t> func_name_2_pos;
static std::vector<std::string> pos_2_func_name;
size_t get_func_name_pos(const char *func_name) {
    return get_pos(func_name, func_name_2_pos, pos_2_func_name);
}

struct Frame {
    size_t file_name_pos;
    size_t func_name_pos;
    int lineno;
};
struct Stacktrace {
    std::vector<Frame> frames;
    std::int64_t duration;
};

static std::vector<std::int64_t> start_times;
static std::vector<Frame> frames;
static std::vector<Stacktrace> stats;

static int profiler_hook(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
    switch (what) {
        case PyTrace_CALL: {
            const char *file_name = nullptr;
            const char *func_name = nullptr;
            int lineno = -1;
            PyObject *code = (PyObject *)PyFrame_GetCode(frame);
            if (code) {
                PyObject *file_name_obj = PyObject_GetAttrString(code, "co_filename");
                if (file_name_obj) {
                    file_name = PyUnicode_AsUTF8(file_name_obj);
                }
                PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
                if (name_obj) {
                    func_name = PyUnicode_AsUTF8(name_obj);
                }
                PyObject *lineno_obj = PyObject_GetAttrString(code, "co_firstlineno");
                if (lineno_obj) {
                    lineno = PyLong_AsLong(lineno_obj);
                }
                Frame frame_info = {
                    get_file_name_pos(file_name),
                    get_func_name_pos(func_name),
                    lineno
                };
                start_times.push_back(std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count());
                frames.push_back(frame_info);
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
            if (frames.empty()) {
                break;
            }
            std::vector<Frame> stacktrace;
            for (size_t i = 0; i < frames.size(); ++i) {
                stacktrace.push_back(frames[i]);
            }
            frames.pop_back();
            std::int64_t duration = end_time - start_time;
            stats.push_back({stacktrace, duration});
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
    PyObject *pos_2_file_name_obj = PyList_New(pos_2_file_name.size());
    for (size_t i = 0; i < pos_2_file_name.size(); ++i) {
        PyList_SetItem(pos_2_file_name_obj, i, PyUnicode_FromString(pos_2_file_name[i].c_str()));
    }
    PyObject *pos_2_func_name_obj = PyList_New(pos_2_func_name.size());
    for (size_t i = 0; i < pos_2_func_name.size(); ++i) {
        PyList_SetItem(pos_2_func_name_obj, i, PyUnicode_FromString(pos_2_func_name[i].c_str()));
    }
    PyObject *stacktraces_obj = PyList_New(stats.size());
    PyObject *durations_obj = PyList_New(stats.size());
    for (size_t i = 0; i < stats.size(); ++i) {
        PyObject *stacktrace_obj = PyList_New(stats[i].frames.size());
        for (size_t j = 0; j < stats[i].frames.size(); ++j) {
            PyObject *frame_info_obj = PyList_New(3);
            PyList_SetItem(frame_info_obj, 0, PyLong_FromLong(stats[i].frames[j].file_name_pos));
            PyList_SetItem(frame_info_obj, 1, PyLong_FromLong(stats[i].frames[j].func_name_pos));
            PyList_SetItem(frame_info_obj, 2, PyLong_FromLong(stats[i].frames[j].lineno));  
            PyList_SetItem(stacktrace_obj, j, frame_info_obj);
        }
        PyList_SetItem(stacktraces_obj, i, stacktrace_obj);
        PyList_SetItem(durations_obj, i, PyLong_FromLong(stats[i].duration));
    }
    PyObject *result = PyDict_New();
    PyDict_SetItem(result, PyUnicode_FromString("pos_2_file_name"), pos_2_file_name_obj);
    PyDict_SetItem(result, PyUnicode_FromString("pos_2_func_name"), pos_2_func_name_obj);
    PyDict_SetItem(result, PyUnicode_FromString("stacktraces"), stacktraces_obj);
    PyDict_SetItem(result, PyUnicode_FromString("durations"), durations_obj);
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