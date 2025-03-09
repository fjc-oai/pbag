#include <pybind11/pybind11.h>
#include <chrono>
#include <cmath>

namespace py = pybind11;

// A function that burns CPU cycles for 'duration' seconds.
void burn(double duration) {
    // Use steady_clock for wall-clock measurement.
    auto start = std::chrono::steady_clock::now();
    volatile double dummy = 0.0;  // volatile to prevent optimization
    while (true) {
        // Do some floating-point arithmetic to burn CPU time.
        dummy += std::sin(dummy) * std::cos(dummy);
        // Check elapsed time.
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start;
        if (elapsed.count() >= duration) {
            break;
        }
    }
    // Use dummy so the compiler cannot optimize away the loop.
    if (dummy == 42.0) {
        throw std::runtime_error("Should not happen");
    }
}

PYBIND11_MODULE(burn_module, m) {
    m.doc() = "Module to burn CPU cycles for a given duration.";
    m.def("burn", &burn, "Burn CPU for a given duration in seconds.",
          py::arg("duration"));
}