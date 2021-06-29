#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zeno/zeno.h>
#include <zeno/GlobalState.h>
#include <zeno/Visualization.h>
namespace py = pybind11;


PYBIND11_MODULE(libzenpy, m) {
    m.def("dumpDescriptors", zeno::dumpDescriptors);
    m.def("bindNodeInput", zeno::bindNodeInput);
    m.def("setNodeParam", zeno::setNodeParam);
    m.def("setNodeOptions", zeno::setNodeOptions);
    m.def("completeNode", zeno::completeNode);
    m.def("switchGraph", zeno::switchGraph);
    m.def("clearNodes", zeno::clearNodes);
    m.def("applyNodes", zeno::applyNodes);
    m.def("addNode", zeno::addNode);

    m.def("setIOPath", [] (std::string const &iopath) {
        zeno::state = zeno::GlobalState();
        return zeno::state.setIOPath(iopath);
    });
    m.def("frameBegin", [] () { return zeno::state.frameBegin(); });
    m.def("substepBegin", [] () { return zeno::state.substepBegin(); });
    m.def("frameEnd", [] () { return zeno::state.frameEnd(); });
    m.def("substepEnd", [] () {
        zeno::Visualization::endFrame();
        return zeno::state.substepEnd();
    });

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (zeno::Exception const &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}
