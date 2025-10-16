#pragma once

#include <zeno/types/IGeometryObject.h>
#include <zeno/types/GeometryObject.h>

namespace zeno {

    struct ObjectRecorder
    {
        std::set<GeometryObject_Adapter*> m_geoms;
        std::set<GeometryObject*> m_geom_impls;
    };
}

