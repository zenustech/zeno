#ifdef ZENO_WITH_PYTHON
#include <Python.h>
#include <zeno/zeno.h>
#include <zeno/utils/log.h>
#include <zeno/utils/string.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#include <zeno/include/zenoutil.h>


namespace zeno {

struct PythonNode : zeno::INode {

    virtual void apply() override {
        auto prim = ZImpl(get_input_prim_param("script"));
        std::string script = zeno::reflect::any_cast<std::string>(prim.result);
        auto& sess = zeno::getSession();
        sess.asyncRunPython(script);
#ifdef _WIN32
        HANDLE hEventPyReady = sess.hEventOfPyFinish();
        WaitForSingleObject(hEventPyReady, INFINITE);
        ResetEvent(hEventPyReady);
#else
#endif
        zany inobj = clone_input("object");
        set_output("object", std::move(inobj));
    }
};

ZENDEFNODE(PythonNode, {
    {
        {gParamType_IObject, "object"},
        {gParamType_String, "script", "", Socket_Primitve, CodeEditor}
    },
    {
        {gParamType_IObject, "object"}
    },
    {},
    {"command"},
});

}

#endif