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

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = (int)std::round(start / time_per_cycle );

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    Alembic::AbcGeom::IPolyMeshSchema::Sample mesamp = mesh.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));

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

    auto &uv0 = prim->tris.add_attr<zeno::vec3f>("uv0");
    auto &uv1 = prim->tris.add_attr<zeno::vec3f>("uv1");
    auto &uv2 = prim->tris.add_attr<zeno::vec3f>("uv2");

    if (auto uv = mesh.getUVsParam()) {
        auto uvsamp = uv.getIndexedValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
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
            log_warn("[alembic] Not found uv, auto fill zero.");
        }
        int count = 0;
        for (auto [start, len]: prim->polys) {
            if (len < 3) continue;
            for (int i = 2; i < len; i++) {
                uv0[count] = zeno::vec3f(0, 0, 0);
                uv1[count] = zeno::vec3f(0, 0, 0);
                uv2[count] = zeno::vec3f(0, 0, 0);
                count += 1;
            }
        }
    }

    return prim;
}

static std::shared_ptr<CameraInfo> foundABCCamera(Alembic::AbcGeom::ICameraSchema &cam, int frameid) {
    CameraInfo cam_info;
    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = cam.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = (int)std::round(start / time_per_cycle );
    int sample_index = clamp(frameid - start_frame, 0, (int)cam.getNumSamples() - 1);

    auto samp = cam.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
    cam_info.focal_length = samp.getFocalLength();
    cam_info._near = samp.getNearClippingPlane();
    cam_info._far = samp.getFarClippingPlane();
    log_info(
        "[alembic] Camera focal_length: {}, near: {}, far: {}",
        cam_info.focal_length,
        cam_info._near,
        cam_info._far
    );
    return std::make_shared<CameraInfo>(cam_info);
}

static Alembic::Abc::v12::M44d foundABCXform(Alembic::AbcGeom::IXformSchema &xfm, int frameid) {
    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = xfm.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = (int)std::round(start / time_per_cycle );
    int sample_index = clamp(frameid - start_frame, 0, (int)xfm.getNumSamples() - 1);

    auto samp = xfm.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
    return samp.getMatrix();
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
        } else if (Alembic::AbcGeom::IXformSchema::matches(md)) {
            if (!read_done) {
                log_info("[alembic] found a Xform [{}]", obj.getName());
            }
            Alembic::AbcGeom::IXform xfm(obj);
            auto &cam_sch = xfm.getSchema();
            tree.xform = foundABCXform(cam_sch, frameid);
        } else if (Alembic::AbcGeom::ICameraSchema::matches(md)) {
            if (!read_done) {
                log_info("[alembic] found a Camera [{}]", obj.getName());
            }
            Alembic::AbcGeom::ICamera cam(obj);
            auto &cam_sch = cam.getSchema();
            tree.camera_info = foundABCCamera(cam_sch, frameid);
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
