#include "Descriptors.h"
#include <QQmlEngine>


static QObject* MySingleTestInstancePtr(QQmlEngine* engine, QJSEngine* scriptEngine)
{
    Q_UNUSED(engine)
    Q_UNUSED(scriptEngine)
    return Descriptors::instance();
}


Descriptors::Descriptors()
{
    initDescs();
    qmlRegisterSingletonType<Descriptors>("Descriptors", 1, 0, "Descriptors", MySingleTestInstancePtr);
}

Descriptors* Descriptors::instance()
{
    static Descriptors __instance;
    return &__instance;
}

NODE_DESCRIPTOR Descriptors::getDescriptor(const QString& name) const
{
    if (m_descs.find(name) == m_descs.end())
        return NODE_DESCRIPTOR();
    return m_descs[name];
}

void Descriptors::initDescs()
{
    m_descs.insert("CreateCube", 
        { "CreateCube", 
        {
            {"position", "vec3f", ParamControl::Vec3edit},
            {"scaleSize", "vec3f", ParamControl::Vec3edit},
            {"rotate", "vec3f", ParamControl::Vec3edit},
            {"hasNormal", "bool", ParamControl::Checkbox},
            {"hasVertUV", "bool", ParamControl::Checkbox},
            {"isFlipFace", "bool", ParamControl::Checkbox},
            {"div_w", "int", ParamControl::Lineddit},
            {"div_h", "int", ParamControl::Lineddit},
            {"div_d", "int", ParamControl::Lineddit},
            {"size", "int", ParamControl::Lineddit},
            {"quads", "bool", ParamControl::Checkbox},
            {"SRC", ""},
        },
        {{"prim", "prim"},
         {"DST", ""}}
    });

    m_descs.insert("ParticlesWrangle",
        { "ParticlesWrangle",
        {
            {"prim", "prim"},
            {"zfxCode", "string", ParamControl::Multiline},
            {"params", "dict"},
            {"SRC", ""}
        },
        {
            {"prim", "prim"},
            {"DST", ""}
        }
        }
    );

    m_descs.insert("SubInput",
        { "SubInput",
            {
                {"defl", ""},
                {"name", "string", ParamControl::Lineddit},
                {"type", "string", ParamControl::Lineddit},
                {"SRC", ""}
            },
            {
                {"port", ""},
                {"hasValue", ""},
                {"DST", ""}
            }
        }
    );

    m_descs.insert("SubOutput",
        {
            "SubOutput",
            {
                {"defl", ""},
                {"name", "string", ParamControl::Lineddit},
                {"type", "string", ParamControl::Lineddit},
                {"port", ""},
                {"SRC", ""}
            },
            {
                {"DST", ""}
            }
        }
    );

    m_descs.insert("NumericInt",
        {
            "NumericInt",
            {
                {"value", "int", ParamControl::Lineddit},
                {"SRC", ""}
            },
            {
                {"value", "int"},
                {"DST", ""}
            }
        }
    );

    m_descs.insert("NumericOperator",
        {
            "NumericOperator",
            {
                {"op_type", "string", ParamControl::Lineddit},
                {"lhs", "int", ParamControl::Lineddit},
                {"rhs", "int", ParamControl::Lineddit},
                {"SRC", ""}
            },
            {
                {"ret", "int"},
                {"DST", ""}
            }
        }
    );

    m_descs.insert("GetFrameNum",
        {
            "GetFrameNum",
        {
            {"SRC", ""}
        },
        {
            {"FrameNum", "int"},
            {"DST", ""}
        }
    });

}