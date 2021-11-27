// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
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

static std::shared_ptr<PrimitiveObject> foundABCMesh(Alembic::AbcGeom::IPolyMeshSchema &mesh) {
    auto prim = std::make_shared<PrimitiveObject>();

    {
        Alembic::AbcGeom::IPolyMeshSchema::Sample mesamp;
        mesh.get(mesamp);

        if (auto marr = mesamp.getPositions()) {
            log_info("[alembic] totally {} positions", marr->size());
            auto &parr = prim->attr<vec3f>("pos");
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
    }

#if 0
    {
        auto nrmpa = mesh.getNormalsParam();
        log_info("normal is indexed: {}", nrmpa.isIndexed());
        auto nrmsp = nrmpa.getIndexedValue();
        auto &nrmind = *nrmsp.getIndices();
        auto &nrmval = *nrmsp.getVals();
        log_info("normal has {} indices, {} values", nrmind.size(), nrmval.size());
        for (size_t i = 0; i < nrmind.size(); i++) {
            auto ind = nrmind[i];
        }
        for (size_t i = 0; i < nrmval.size(); i++) {
            auto val = nrmval[i];
        }
    }

    {
        auto uvpa = mesh.getUVsParam();
        log_info("UV is indexed: {}", uvpa.isIndexed());
        auto uvsp = uvpa.getIndexedValue();
        auto &uvind = *uvsp.getIndices();
        auto &uvval = *uvsp.getVals();
        log_info("UV has {} indices, {} values", uvind.size(), uvval.size());
        for (size_t i = 0; i < uvind.size(); i++) {
            auto ind = uvind[i];
        }
        for (size_t i = 0; i < uvval.size(); i++) {
            auto val = uvval[i];
        }
    }
#endif
}

struct ABCTree {
    std::shared_ptr<PrimitiveObject> prim;
    std::vector<std::unique_ptr<ABCTree>> children;

    inline std::shared_ptr<PrimitiveObject> getFirstPrim() const {
        if (prim) return prim;
        for (auto const &ch: children)
            if (auto p = ch->getFirstPrim())
                return p;
        return nullptr;
    }
};

static void traverseABC
( Alembic::AbcGeom::IObject &obj
, ABCTree &tree
) {
    {
        auto const &md = obj.getMetaData();
        log_info("[alembic] meta data: [{}]", md.serialize());

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
        decltype(auto) name = obj.getChildHeader(i).getName();
        log_info("[alembic] at {} name: [{}]", i, name);

        Alembic::AbcGeom::IObject child(obj, name);

        auto childTree = std::make_unique<ABCTree>();
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
        auto abctree = std::make_unique<ABCTree>();
        {
            auto path = get_input<StringObject>("path")->get();
            auto archive = readABC(path);
            auto obj = archive.getTop();
            traverseABC(obj, *abctree);
        }
        auto retprim = abctree->getFirstPrim();
        set_output("prim", std::move(retprim));
    }
};

ZENDEFNODE(ReadAlembic, {
    {{"string", "path"}},
    {{"PrimitiveObject", "prim"}},
    {},
    {"alembic"},
});

} // namespace zeno
