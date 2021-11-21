#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <zeno/zeno.h>
#ifdef ZENO_GLOBALSTATE
#include <zeno/extra/GlobalState.h>
#endif
#ifdef ZENO_VISUALIZATION
#include <zeno/extra/Visualization.h>
#endif
#ifdef ZENO_FAULTHANDLER
#include <zeno/extra/FaultHandler.h>
#endif
namespace py = pybind11;


PYBIND11_MODULE(pylib_zeno, m) {
    m.def("dumpDescriptors", zeno::dumpDescriptors);
    m.def("bindNodeInput", zeno::bindNodeInput);
    m.def("setNodeParam", zeno::setNodeParam);
    m.def("setNodeOption", zeno::setNodeOption);
    m.def("clearAllState", zeno::clearAllState);
    m.def("completeNode", zeno::completeNode);
    m.def("switchGraph", zeno::switchGraph);
    m.def("clearNodes", zeno::clearNodes);
    m.def("applyNodes", zeno::applyNodes);
    m.def("loadScene", zeno::loadScene);
    m.def("addNode", zeno::addNode);

#ifdef ZENO_GLOBALSTATE
    m.def("setIOPath", [] (std::string const &iopath) {
        zeno::state = zeno::GlobalState();
        return zeno::state.setIOPath(iopath);
    });
    m.def("substepBegin", [] () { return zeno::state.substepBegin(); });
    m.def("substepEnd", [] () { return zeno::state.substepEnd(); });
    m.def("frameBegin", [] () { return zeno::state.frameBegin(); });
    m.def("frameEnd", [] () {
#ifdef ZENO_VISUALIZATION
        zeno::Visualization::endFrame();
#endif
        return zeno::state.frameEnd();
    });
#endif

    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p)
                std::rethrow_exception(p);
        } catch (zeno::BaseException const &e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
    });
}
