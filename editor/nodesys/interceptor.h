#ifndef INTERCEPTOR_H
#define INTERCEPTOR_H

#include "qdmgraphicsnode.h"
#include "qdmgraphicsscene.h"
#include "qdmgraphicslinkfull.h"
#include <zeno/dop/SceneGraph.h>

ZENO_NAMESPACE_BEGIN

struct Interceptor {
    static void toDopGraph
    ( QDMGraphicsScene *scene
    , dop::SceneGraph *d_scene
    );
};

ZENO_NAMESPACE_END

#endif // INTERCEPTOR_H
