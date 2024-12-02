#include <pybind11/pybind11.h>

#include <chrono>
int busy_loop(int dur_ms) {
    auto start = std::chrono::high_resolution_clock::now();

    while (true) {
        auto now = std::chrono::high_resolution_clock::now();
        if (std::chrono::duration_cast<std::chrono::milliseconds>(now - start).count() >= dur_ms) {
            break;
        }
    }
}

namespace py = pybind11;

PYBIND11_MODULE(ops, m) {
    m.def("busy_loop", &busy_loop, "A function that adds two numbers");
}