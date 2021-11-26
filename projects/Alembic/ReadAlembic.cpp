#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/PrimitiveObject.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/Abc/ErrorHandler.h>

namespace zeno {

static void traverseABC(Alembic::AbcGeom::IObject &obj)
{
    size_t nch = obj.getNumChildren();
    log_info("found {} children", nch);

    for (size_t i = 0; i < nch; i++) {
        decltype(auto) name = obj.getChildHandler(i).getName();
        log_info("at {} name: [{}]", i, name);

        Alembic::AbcGeom::IObject child(obj, name);

        auto const &md = child.getMetaData();
        log_info("meta data: [{}]", md.serialize());

        if ( Alembic::AbcGeom::IPolyMeshSchema::matches(md)
          || Alembic::AbcGeom::ISubDSchema::matches(md)
           ) {
            log_info("found a mesh [{}]", child.getName());
        }

        traverseABC(child);
    }
}

static void readABC(std::string const &path) {
    Alembic::AbcGeom::IArchive archive
        ( Alembic::AbcCoreOgawa::ReadArchive()
        , path
        );
    Alembic::AbcGeom::IObject obj = archive.getTop();
    traverseABC(obj);
}

struct ReadAlembic {
    virtual void apply() override {
        readABC("/tmp/a.obj");
    }
};

ZENDEFNODE(ReadAlembic, {
    {{"string", "path"}},
    {{"PrimitiveObject", "prim"},
    {},
    {"alembic"},
});

} // namespace zeno
