// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/extra/GlobalState.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include "ABCTree.h"
#include <cstring>
#include <cstdio>

namespace zeno {
namespace {

static std::shared_ptr<PrimitiveObject> foundABCMesh(Alembic::AbcGeom::IPolyMeshSchema &mesh) {
    auto prim = std::make_shared<PrimitiveObject>();
    int i = zeno::state.frameid;
    i = std::min((int)i, (int)mesh.getNumSamples() - 1);

    Alembic::AbcGeom::IPolyMeshSchema::Sample mesamp = mesh.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)i));

    if (auto marr = mesamp.getPositions()) {
        log_info("[alembic] totally {} positions", marr->size());
        auto &parr = prim->verts;
        for (size_t i = 0; i < marr->size(); i++) {
            auto const &val = (*marr)[i];
            parr.emplace_back(val[0], val[1], val[2]);
        }
    }

    if (auto marr = mesamp.getVelocities()) {
        log_info("[alembic] totally {} velocities", marr->size());
        auto &parr = prim->attr<vec3f>("vel");
        for (size_t i = 0; i < marr->size(); i++) {
            auto const &val = (*marr)[i];
            parr.emplace_back(val[0], val[1], val[2]);
        }
    }

    if (auto marr = mesamp.getFaceCounts()) {
        log_info("[alembic] totally {} faces", marr->size());
        auto &parr = prim->polys;
        int base = 0;
        for (size_t i = 0; i < marr->size(); i++) {
            int cnt = (*marr)[i];
            parr.emplace_back(base, cnt);
            base += cnt;
        }
    }

    if (auto marr = mesamp.getFaceIndices()) {
        log_info("[alembic] totally {} face indices", marr->size());
        auto &parr = prim->loops;
        for (size_t i = 0; i < marr->size(); i++) {
            int ind = (*marr)[i];
            parr.push_back(ind);
        }
    }

    return prim;
}

static void traverseABC
( Alembic::AbcGeom::IObject &obj
, ABCTree &tree
) {
    {
        auto const &md = obj.getMetaData();
        log_info("[alembic] meta data: [{}]", md.serialize());
        tree.name = obj.getName();

        if (Alembic::AbcGeom::IPolyMesh::matches(md)) {
            log_info("[alembic] found a mesh [{}]", obj.getName());

            Alembic::AbcGeom::IPolyMesh meshy(obj);
            auto &mesh = meshy.getSchema();
            tree.prim = foundABCMesh(mesh);
        }
    }

    size_t nch = obj.getNumChildren();
    log_info("[alembic] found {} children", nch);

    for (size_t i = 0; i < nch; i++) {
        auto const &name = obj.getChildHeader(i).getName();
        log_info("[alembic] at {} name: [{}]", i, name);

        Alembic::AbcGeom::IObject child(obj, name);

        auto childTree = std::make_shared<ABCTree>();
        traverseABC(child, *childTree);
        tree.children.push_back(std::move(childTree));
    }
}

static Alembic::AbcGeom::IArchive readABC(std::string const &path) {
    std::string hdr;
    {
        char buf[5];
        std::memset(buf, 0, 5);
        auto fp = std::fopen(path.c_str(), "rb");
        if (!fp)
            throw Exception("[alembic] cannot open file for read: " + path);
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
        throw Exception("[alembic] unrecognized ABC header: [" + hdr + "]");
    }
}

struct ReadAlembic : INode {
    virtual void apply() override {
        auto abctree = std::make_shared<ABCTree>();
        {
            auto path = get_input<StringObject>("path")->get();
            auto archive = readABC(path);
            auto obj = archive.getTop();
            traverseABC(obj, *abctree);
        }
        set_output("abctree", std::move(abctree));
    }
};

ZENDEFNODE(ReadAlembic, {
    {{"readpath", "path"}},
    {{"ABCTree", "abctree"}},
    {},
    {"alembic"},
});

} // namespace
} // namespace zeno
