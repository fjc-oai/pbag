#include <Python.h>
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <string_view>



static std::unordered_map<std::string_view, size_t> file_name_2_pos;
static std::vector<std::string> pos_2_file_name;
static std::unordered_map<std::string_view, size_t> func_name_2_pos;
static std::vector<std::string> pos_2_func_name;

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

size_t get_func_name_pos(const char *func_name) {
    return get_pos(func_name, func_name_2_pos, pos_2_func_name);
}

struct FrameInfo {
    size_t file_name_pos;
    size_t func_name_pos;
    int lineno;
    std::int64_t st;
    std::int64_t ed;
};

struct CallStack {
    std::vector<FrameInfo> frames;
};

struct TraceStats {
    std::vector<FrameInfo> frames;
};

using ThreadId = unsigned long;
static std::unordered_map<ThreadId, CallStack> threadCallStack;
static std::unordered_map<ThreadId, TraceStats> threadTraceStats;

static int profiler_hook(PyObject *obj, PyFrameObject *frame, int what, PyObject *arg) {
    switch (what) {
        case PyTrace_CALL: {
            PyObject *code = (PyObject *)PyFrame_GetCode(frame);
            if (code) {
                const ThreadId threadId = PyThread_get_thread_ident();
                const char *file_name = nullptr;
                const char *func_name = nullptr;
                int lineno = -1;
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
                const auto current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::steady_clock::now().time_since_epoch()
                ).count();
                FrameInfo frame_info = FrameInfo{
                    get_file_name_pos(file_name),
                    get_func_name_pos(func_name),
                    lineno,
                    current_time,
                    0,
                };
                threadCallStack[threadId].frames.push_back(std::move(frame_info));
            }
            break;
        }
        case PyTrace_RETURN: {
            const ThreadId threadId = PyThread_get_thread_ident();
            auto& callstack = threadCallStack[threadId];
            if (callstack.frames.empty()) { // this is expected when there is function calls before set_profiler but returned before unset_profiler
                bool DEBUG_PRINT = false;
                if (DEBUG_PRINT) {
                    printf("No callstack for thread %lu\n", threadId); 
                    PyObject *code = (PyObject *)PyFrame_GetCode(frame);
                    const char *func_name = nullptr;
                    const char *file_name = nullptr;
                    int lineno = -1;
                    PyObject *name_obj = PyObject_GetAttrString(code, "co_name");
                    if (name_obj) {
                        func_name = PyUnicode_AsUTF8(name_obj);
                    }
                    printf("func_name: %s\n", func_name);
                    PyObject *file_name_obj = PyObject_GetAttrString(code, "co_filename");
                    if (file_name_obj) {
                        file_name = PyUnicode_AsUTF8(file_name_obj);
                    }
                    printf("file_name: %s\n", file_name);
                    PyObject *lineno_obj = PyObject_GetAttrString(code, "co_firstlineno");
                    if (lineno_obj) {
                        lineno = PyLong_AsLong(lineno_obj);
                    }
                    printf("lineno: %d\n", lineno);
                }
                break;
            }
            auto last_frame = callstack.frames.back(); // Perf might be optimized
            const auto current_time = std::chrono::duration_cast<std::chrono::microseconds>(
                std::chrono::steady_clock::now().time_since_epoch()
            ).count();
            last_frame.ed = current_time;
            callstack.frames.pop_back();
            threadTraceStats[threadId].frames.push_back(std::move(last_frame));
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
    PyObject *thread_stats_obj = PyDict_New();
    for (const auto& [threadId, stats] : threadTraceStats) {
        PyObject *stats_obj = PyList_New(stats.frames.size());
        for (size_t i = 0; i < stats.frames.size(); ++i) {
            PyObject *frame_info_obj = PyList_New(5);
            PyList_SetItem(frame_info_obj, 0, PyLong_FromLong(stats.frames[i].file_name_pos));
            PyList_SetItem(frame_info_obj, 1, PyLong_FromLong(stats.frames[i].func_name_pos));
            PyList_SetItem(frame_info_obj, 2, PyLong_FromLong(stats.frames[i].lineno));
            PyList_SetItem(frame_info_obj, 3, PyLong_FromLong(stats.frames[i].st));
            PyList_SetItem(frame_info_obj, 4, PyLong_FromLong(stats.frames[i].ed));
            PyList_SetItem(stats_obj, i, frame_info_obj);
        }
        PyDict_SetItem(thread_stats_obj, PyLong_FromUnsignedLong(threadId), stats_obj);
    }

    PyObject *pos_2_file_name_obj = PyList_New(pos_2_file_name.size());
    for (size_t i = 0; i < pos_2_file_name.size(); ++i) {
        PyList_SetItem(pos_2_file_name_obj, i, PyUnicode_FromString(pos_2_file_name[i].c_str()));
    }
    PyObject *pos_2_func_name_obj = PyList_New(pos_2_func_name.size());
    for (size_t i = 0; i < pos_2_func_name.size(); ++i) {
        PyList_SetItem(pos_2_func_name_obj, i, PyUnicode_FromString(pos_2_func_name[i].c_str()));
    }
    PyDict_SetItem(result, PyUnicode_FromString("thread_stats"), thread_stats_obj);
    PyDict_SetItem(result, PyUnicode_FromString("pos_2_file_name"), pos_2_file_name_obj);
    PyDict_SetItem(result, PyUnicode_FromString("pos_2_func_name"), pos_2_func_name_obj);
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