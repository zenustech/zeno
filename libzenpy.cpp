#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zeno/zen.h>
#include <zeno/GlobalState.h>
#include <zeno/Visualization.h>
namespace py = pybind11;


PYBIND11_MODULE(libzenpy, m) {
    m.def("dumpDescriptors", zen::dumpDescriptors);
    m.def("bindNodeInput", zen::bindNodeInput);
    m.def("setNodeParam", zen::setNodeParam);
    m.def("setNodeOptions", zen::setNodeOptions);
    m.def("completeNode", zen::completeNode);
    m.def("switchGraph", zen::switchGraph);
    m.def("clearNodes", zen::clearNodes);
    m.def("applyNodes", zen::applyNodes);
    m.def("addNode", zen::addNode);

    m.def("setIOPath", [] (std::string const &iopath) {
        zen::state = zen::GlobalState();
        return zen::state.setIOPath(iopath);
    });
    m.def("frameBegin", [] () { return zen::state.frameBegin(); });
    m.def("frameEnd", [] () {
        zen::Visualization::endFrame();
        return zen::state.frameEnd();
    });
    m.def("substepBegin", [] () { return zen::state.substepBegin(); });
    m.def("substepEnd", [] () { return zen::state.substepEnd(); });

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (zen::Exception const &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}
