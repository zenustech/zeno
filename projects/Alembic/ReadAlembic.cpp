#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/StringObject.h>
#include <zeno/PrimitiveObject.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include <cstdio>
#include <cstring>
#include <optional>

namespace zeno {

static void traverseABC(Alembic::AbcGeom::IObject &obj)
{
    size_t nch = obj.getNumChildren();
    log_info("[alembic] found {} children", nch);

    for (size_t i = 0; i < nch; i++) {
        decltype(auto) name = obj.getChildHeader(i).getName();
        log_info("[alembic] at {} name: [{}]", i, name);

        Alembic::AbcGeom::IObject child(obj, name);

        auto const &md = child.getMetaData();
        log_info("[alembic] meta data: [{}]", md.serialize());

        if ( Alembic::AbcGeom::IPolyMeshSchema::matches(md)
          || Alembic::AbcGeom::ISubDSchema::matches(md)
           ) {
            log_info("[alembic] found a mesh [{}]", child.getName());
        }

        traverseABC(child);
    }
}

static Alembic::AbcGeom::IArchive readABC(std::string const &path) {
    std::string hdr;
    {
        char buf[5];
        std::memset(buf, 0, 5);
        auto fp = std::fopen(path.c_str(), "rb");
        if (!fp)
            throw zeno::Exception("[alembic] cannot open file for read: " + path);
        std::fread(buf, 4, 1, fp);
        std::fclose(fp);
        hdr = buf;
    }
    if (hdr == "\x89HDF") {
        log_info("[alembic] opening as HDF5 format");
        return {Alembic::AbcCoreHDF5::ReadArchive(), path};
    } else if (hdr == "Ogaw") {
        log_info("[alembic] opening as Ogawa format");
        return {Alembic::AbcCoreOgawa::ReadArchive(), path};
    } else {
        throw zeno::Exception("[alembic] unrecognized ABC header: [" + hdr + "]");
    }
}

struct ReadAlembic : zeno::INode {
    virtual void apply() override {
        auto path = get_input<StringObject>("path")->get();
        auto archive = readABC(path);
        auto obj = archive.getTop();
        traverseABC(obj);
    }
};

ZENDEFNODE(ReadAlembic, {
    {{"string", "path"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

} // namespace zeno
