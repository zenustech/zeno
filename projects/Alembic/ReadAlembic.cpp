// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
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

static int clamp(int i, int _min, int _max) {
    if (i < _min) {
        return _min;
    } else if (i > _max) {
        return _max;
    } else {
        return i;
    }
}

static std::shared_ptr<PrimitiveObject> foundABCMesh(Alembic::AbcGeom::IPolyMeshSchema &mesh, int frameid, bool read_done) {
    auto prim = std::make_shared<PrimitiveObject>();

    frameid = clamp(frameid, 0, (int)mesh.getNumSamples() - 1);
    Alembic::AbcGeom::IPolyMeshSchema::Sample mesamp = mesh.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)frameid));

    if (auto marr = mesamp.getPositions()) {
        if (!read_done) {
            log_info("[alembic] totally {} positions", marr->size());
        }
        auto &parr = prim->verts;
        for (size_t i = 0; i < marr->size(); i++) {
            auto const &val = (*marr)[i];
            parr.emplace_back(val[0], val[1], val[2]);
        }
    }

    if (auto marr = mesamp.getVelocities()) {
        if (!read_done) {
            log_info("[alembic] totally {} velocities", marr->size());
        }
        auto &parr = prim->attr<vec3f>("vel");
        for (size_t i = 0; i < marr->size(); i++) {
            auto const &val = (*marr)[i];
            parr.emplace_back(val[0], val[1], val[2]);
        }
    }

    if (auto marr = mesamp.getFaceCounts()) {
        if (!read_done) {
            log_info("[alembic] totally {} faces", marr->size());
        }
        auto &parr = prim->polys;
        int base = 0;
        for (size_t i = 0; i < marr->size(); i++) {
            int cnt = (*marr)[i];
            parr.emplace_back(base, cnt);
            base += cnt;
        }
    }

    if (auto marr = mesamp.getFaceIndices()) {
        if (!read_done) {
            log_info("[alembic] totally {} face indices", marr->size());
        }
        auto &parr = prim->loops;
        for (size_t i = 0; i < marr->size(); i++) {
            int ind = (*marr)[i];
            parr.push_back(ind);
        }
    }

    prim_triangulate(prim.get());

    if (auto uv = mesh.getUVsParam()) {
        auto uvsamp = uv.getIndexedValue();
        int value_size = (int) uvsamp.getVals()->size();
        int index_size = (int) uvsamp.getIndices()->size();
        if (!read_done) {
            log_info("[alembic] totally {} uv value", value_size);
            log_info("[alembic] totally {} uv indices", index_size);
            if (prim->loops.size() == index_size) {
                log_info("[alembic] uv per face");
            } else if (prim->verts.size() == index_size) {
                log_info("[alembic] uv per vertex");
            } else {
                log_error("[alembic] error uv indices");
            }
        }
        auto uv_value = std::vector<zeno::vec3f>();
        {
            auto marr = uvsamp.getVals();
            for (size_t i = 0; i < marr->size(); i++) {
                auto const &val = (*marr)[i];
                uv_value.push_back(zeno::vec3f(val[0], val[1], 0));
            }
        }
        auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
        auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
        auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");
        auto uv_loops = std::vector<int>();
        std::vector<int> *uv_loops_ref;
        if (prim->loops.size() == index_size) {
            int start = 0;
            {
                auto marr = uvsamp.getIndices();
                for (size_t i = 0; i < marr->size(); i++) {
                    int idx = (*marr)[i];
                    uv_loops.push_back(idx);
                }
            }
            uv_loops_ref = &uv_loops;
        } else if (prim->verts.size() == index_size) {
            uv_loops_ref = &(prim->loops.values);
        }
        int count = 0;
        for (auto [start, len]: prim->polys) {
            if (len < 3) continue;
            for (int i = 2; i < len; i++) {
                uv0[count] = uv_value[(*uv_loops_ref)[start]];
                uv1[count] = uv_value[(*uv_loops_ref)[start + i - 1]];
                uv2[count] = uv_value[(*uv_loops_ref)[start + i]];
                count += 1;
            }
        }
    } else {
        if (!read_done) {
            log_info("[alembic] Not found uv");
        }
    }

    return prim;
}

static void traverseABC(
    Alembic::AbcGeom::IObject &obj,
    ABCTree &tree,
    int frameid,
    bool read_done
) {
    {
        auto const &md = obj.getMetaData();
        if (!read_done) {
            log_info("[alembic] meta data: [{}]", md.serialize());
        }
        tree.name = obj.getName();

        if (Alembic::AbcGeom::IPolyMesh::matches(md)) {
            if (!read_done) {
                log_info("[alembic] found a mesh [{}]", obj.getName());
            }

            Alembic::AbcGeom::IPolyMesh meshy(obj);
            auto &mesh = meshy.getSchema();
            tree.prim = foundABCMesh(mesh, frameid, read_done);
        }
    }

    size_t nch = obj.getNumChildren();
    if (!read_done) {
        log_info("[alembic] found {} children", nch);
    }

    for (size_t i = 0; i < nch; i++) {
        auto const &name = obj.getChildHeader(i).getName();
        if (!read_done) {
            log_info("[alembic] at {} name: [{}]", i, name);
        }

        Alembic::AbcGeom::IObject child(obj, name);

        auto childTree = std::make_shared<ABCTree>();
        traverseABC(child, *childTree, frameid, read_done);
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
    Alembic::Abc::v12::IArchive archive;
    bool read_done = false;
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<NumericObject>("frameid")->get<int>();
        } else {
            frameid = zeno::state.frameid;
        }
        auto abctree = std::make_shared<ABCTree>();
        {
            auto path = get_input<StringObject>("path")->get();
            if (read_done == false) {
                archive = readABC(path);
            }
            double start, _end;
            GetArchiveStartAndEndTime(archive, start, _end);
            // fmt::print("GetArchiveStartAndEndTime: {}\n", start);
            // fmt::print("archive.getNumTimeSamplings: {}\n", archive.getNumTimeSamplings());
            auto obj = archive.getTop();
            traverseABC(obj, *abctree, frameid, read_done);
            read_done = true;
        }
        set_output("abctree", std::move(abctree));
    }
};

ZENDEFNODE(ReadAlembic, {
    {{"readpath", "path"}, {"frameid"}},
    {{"ABCTree", "abctree"}},
    {},
    {"alembic"},
});

} // namespace
} // namespace zeno
