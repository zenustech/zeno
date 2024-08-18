#include "descriptors.h"
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
            {"position", "vec3f", zeno::Vec3edit},
            {"scaleSize", "vec3f", zeno::Vec3edit},
            {"rotate", "vec3f", zeno::Vec3edit},
            {"hasNormal", "bool", zeno::Checkbox},
            {"hasVertUV", "bool", zeno::Checkbox},
            {"isFlipFace", "bool", zeno::Checkbox},
            {"div_w", "int", zeno::Lineedit},
            {"div_h", "int", zeno::Lineedit},
            {"div_d", "int", zeno::Lineedit},
            {"size", "int", zeno::Lineedit},
            {"quads", "bool", zeno::Checkbox},
            {"SRC", ""},
        },
        {{gParamType_Primitive, "prim"},
         {"DST", ""}}
    });

    m_descs.insert("ParticlesWrangle",
        { "ParticlesWrangle",
        {
            {gParamType_Primitive, "prim"},
            {"zfxCode", "string", zeno::Multiline},
            {"params", "dict"},
            {"SRC", ""}
        },
        {
            {gParamType_Primitive, "prim"},
            {"DST", ""}
        }
        }
    );

    m_descs.insert("SubInput",
        { "SubInput",
            {
                {"defl", ""},
                {"name", "string", zeno::Lineedit},
                {"type", "string", zeno::Lineedit},
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
                {"name", "string", zeno::Lineedit},
                {"type", "string", zeno::Lineedit},
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
                {"value", "int", zeno::Lineedit},
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
                {"op_type", "string", zeno::Lineedit},
                {"lhs", "int", zeno::Lineedit},
                {"rhs", "int", zeno::Lineedit},
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