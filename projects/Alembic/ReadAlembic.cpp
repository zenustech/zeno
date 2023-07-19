// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/extra/GlobalState.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include "ABCTree.h"
#include <cstring>
#include <cstdio>
#include <filesystem>

using namespace Alembic::AbcGeom;

namespace zeno {
static int clamp(int i, int _min, int _max) {
    if (i < _min) {
        return _min;
    } else if (i > _max) {
        return _max;
    } else {
        return i;
    }
}

static void read_velocity(std::shared_ptr<PrimitiveObject> prim, V3fArraySamplePtr marr, bool read_done) {
    if (marr == nullptr) {
        return;
    }
    if (marr->size() > 0) {
        if (!read_done) {
            log_info("[alembic] totally {} velocities", marr->size());
        }
        auto &parr = prim->add_attr<vec3f>("v");
        for (size_t i = 0; i < marr->size(); i++) {
            auto const &val = (*marr)[i];
            parr[i] = {val[0], val[1], val[2]};
        }
    }
}

static void read_attributes(std::shared_ptr<PrimitiveObject> prim, ICompoundProperty arbattrs, const ISampleSelector &iSS, bool read_done) {
    if (!arbattrs) {
        return;
    }
    size_t numProps = arbattrs.getNumProperties();
    for (auto i = 0; i < numProps; i++) {
        PropertyHeader p = arbattrs.getPropertyHeader(i);
        if (IFloatGeomParam::matches(p)) {
            IFloatGeomParam param(arbattrs, p.getName());

            IFloatGeomParam::Sample samp = param.getIndexedValue(iSS);
            std::vector<float> data;
            data.resize(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                data[i] = samp.getVals()->get()[i];
            }
            if (!read_done) {
                log_info("[alembic] float attr {}, len {}.", p.getName(), data.size());
            }

            if (prim->verts.size() == data.size()) {
                auto &attr = prim->add_attr<float>(p.getName());
                for (auto i = 0; i < prim->verts.size(); i++) {
                    attr[i] = data[i];
                }
            }
            else if (prim->verts.size() * 3 == data.size()) {
                auto &attr = prim->add_attr<zeno::vec3f>(p.getName());
                for (auto i = 0; i < prim->verts.size(); i++) {
                    attr[i] = { data[ 3 * i], data[3 * i + 1], data[3 * i + 2]};
                }
            }
            else {
                if (!read_done) {
                    log_error("[alembic] can not load attr {}. Check if link to Points channel when exported from Houdini.", p.getName());
                }
            }
        }
        else if (IV3fGeomParam::matches(p)) {
            IV3fGeomParam param(arbattrs, p.getName());
            if (!read_done) {
                log_info("[alembic] vec3f attr {}.", p.getName());
            }
            IV3fGeomParam::Sample samp = param.getIndexedValue(iSS);
            if (prim->verts.size() == samp.getVals()->size()) {
                auto &attr = prim->add_attr<zeno::vec3f>(p.getName());
                for (auto i = 0; i < prim->verts.size(); i++) {
                    auto v = samp.getVals()->get()[i];
                    attr[i] = {v[0], v[1], v[2]};
                }
            }
        }
        else if (IN3fGeomParam::matches(p)) {
            if (!read_done) {
                log_info("[alembic] IN3fGeomParam attr {}.", p.getName());
            }
            IN3fGeomParam param(arbattrs, p.getName());
            IN3fGeomParam::Sample samp = param.getIndexedValue(iSS);
            if (prim->verts.size() == samp.getVals()->size()) {
                auto &attr = prim->add_attr<zeno::vec3f>(p.getName());
                for (auto i = 0; i < prim->verts.size(); i++) {
                    auto v = samp.getVals()->get()[i];
                    attr[i] = {v[0], v[1], v[2]};
                }
            }
        }
        else if (IC3fGeomParam::matches(p)) {
            if (!read_done) {
                log_info("[alembic] IC3fGeomParam attr {}.", p.getName());
            }
            IC3fGeomParam param(arbattrs, p.getName());
            IC3fGeomParam::Sample samp = param.getIndexedValue(iSS);
            if (prim->verts.size() == samp.getVals()->size()) {
                auto &attr = prim->add_attr<zeno::vec3f>(p.getName());
                for (auto i = 0; i < prim->verts.size(); i++) {
                    auto v = samp.getVals()->get()[i];
                    attr[i] = {v[0], v[1], v[2]};
                }
            }
        }
        else {
            if (!read_done) {
                log_error("[alembic] can not load attr {}..", p.getName());
            }
        }
    }
}

static void read_user_data(std::shared_ptr<PrimitiveObject> prim, ICompoundProperty arbattrs, const ISampleSelector &iSS, bool read_done) {
    if (!arbattrs) {
        return;
    }
    size_t numProps = arbattrs.getNumProperties();
    for (auto i = 0; i < numProps; i++) {
        PropertyHeader p = arbattrs.getPropertyHeader(i);
        if (IFloatProperty::matches(p)) {
            IFloatProperty param(arbattrs, p.getName());

            float v = param.getValue(iSS);
            prim->userData().set2(p.getName(), v);
        }
        else if (IInt32Property::matches(p)) {
            IInt32Property param(arbattrs, p.getName());

            int v = param.getValue(iSS);
            prim->userData().set2(p.getName(), v);
        }
        else if (IV2fProperty::matches(p)) {
            IV2fProperty param(arbattrs, p.getName());

            auto v = param.getValue(iSS);
            prim->userData().set2(p.getName(), vec2f(v[0], v[1]));
        }
        else if (IV3fProperty::matches(p)) {
            IV3fProperty param(arbattrs, p.getName());

            auto v = param.getValue(iSS);
            prim->userData().set2(p.getName(), vec3f(v[0], v[1], v[2]));
        }
        else if (IV2iProperty::matches(p)) {
            IV2iProperty param(arbattrs, p.getName());

            auto v = param.getValue(iSS);
            prim->userData().set2(p.getName(), vec2i(v[0], v[1]));
        }
        else if (IV3iProperty::matches(p)) {
            IV3iProperty param(arbattrs, p.getName());

            auto v = param.getValue(iSS);
            prim->userData().set2(p.getName(), vec3i(v[0], v[1], v[2]));
        }
        else if (IStringProperty::matches(p)) {
            IStringProperty param(arbattrs, p.getName());

            auto value = param.getValue(iSS);
            prim->userData().set2(p.getName(), value);
        }
        else {
            if (!read_done) {
                log_error("[alembic] can not load user data {}..", p.getName());
            }
        }
    }
}

static std::shared_ptr<PrimitiveObject> foundABCMesh(Alembic::AbcGeom::IPolyMeshSchema &mesh, int frameid, bool read_done) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = (int)std::round(start / time_per_cycle );

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    ISampleSelector iSS = Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index);
    Alembic::AbcGeom::IPolyMeshSchema::Sample mesamp = mesh.getValue(iSS);

    if (auto marr = mesamp.getPositions()) {
        if (!read_done) {
            log_debug("[alembic] totally {} positions", marr->size());
        }
        auto &parr = prim->verts;
        for (size_t i = 0; i < marr->size(); i++) {
            auto const &val = (*marr)[i];
            parr.emplace_back(val[0], val[1], val[2]);
        }
    }

    read_velocity(prim, mesamp.getVelocities(), read_done);
    if (auto nrm = mesh.getNormalsParam()) {
        auto nrmsamp =
                nrm.getIndexedValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
        int value_size = (int)nrmsamp.getVals()->size();
        if (value_size == prim->verts.size()) {
            auto &nrms = prim->verts.add_attr<vec3f>("nrm");
            auto marr = nrmsamp.getVals();
            for (size_t i = 0; i < marr->size(); i++) {
                auto const &n = (*marr)[i];
                nrms[i] = {n[0], n[1], n[2]};
            }
        }
    }

    if (auto marr = mesamp.getFaceIndices()) {
        if (!read_done) {
            log_debug("[alembic] totally {} face indices", marr->size());
        }
        auto &parr = prim->loops;
        for (size_t i = 0; i < marr->size(); i++) {
            int ind = (*marr)[i];
            parr.push_back(ind);
        }
    }

    if (auto marr = mesamp.getFaceCounts()) {
        if (!read_done) {
            log_debug("[alembic] totally {} faces", marr->size());
        }
        auto &loops = prim->loops;
        auto &parr = prim->polys;
        int base = 0;
        for (size_t i = 0; i < marr->size(); i++) {
            int cnt = (*marr)[i];
            parr.emplace_back(base, cnt);
            base += cnt;
        }
    }
    if (auto uv = mesh.getUVsParam()) {
        auto uvsamp =
            uv.getIndexedValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
        int value_size = (int)uvsamp.getVals()->size();
        int index_size = (int)uvsamp.getIndices()->size();
        if (!read_done) {
            log_debug("[alembic] totally {} uv value", value_size);
            log_debug("[alembic] totally {} uv indices", index_size);
            if (prim->loops.size() == index_size) {
                log_debug("[alembic] uv per face");
            } else if (prim->verts.size() == index_size) {
                log_debug("[alembic] uv per vertex");
            } else {
                log_error("[alembic] error uv indices");
            }
        }
        prim->uvs.resize(value_size);
        {
            auto marr = uvsamp.getVals();
            for (size_t i = 0; i < marr->size(); i++) {
                auto const &val = (*marr)[i];
                prim->uvs[i] = {val[0], val[1]};
            }
        }
        if (prim->loops.size() == index_size) {
            prim->loops.add_attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                prim->loops.attr<int>("uvs")[i] = (*uvsamp.getIndices())[i];
            }
        }
        else if (prim->verts.size() == index_size) {
            prim->loops.add_attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                prim->loops.attr<int>("uvs")[i] = prim->loops[i];
            }
        }
    }
    if (!prim->loops.has_attr("uvs")) {
        if (!read_done) {
            log_warn("[alembic] Not found uv, auto fill zero.");
        }
        prim->uvs.resize(1);
        prim->uvs[0] = zeno::vec2f(0, 0);
        prim->loops.add_attr<int>("uvs");
        for (auto i = 0; i < prim->loops.size(); i++) {
            prim->loops.attr<int>("uvs")[i] = 0;
        }
    }
    ICompoundProperty arbattrs = mesh.getArbGeomParams();
    read_attributes(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = mesh.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);

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

static std::shared_ptr<PrimitiveObject> foundABCPoints(Alembic::AbcGeom::IPointsSchema &mesh, int frameid, bool read_done) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = (int)std::round(start / time_per_cycle );

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    auto iSS = Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index);
    Alembic::AbcGeom::IPointsSchema::Sample mesamp = mesh.getValue(iSS);
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
    read_velocity(prim, mesamp.getVelocities(), read_done);
    ICompoundProperty arbattrs = mesh.getArbGeomParams();
    read_attributes(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = mesh.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);
    return prim;
}

static std::shared_ptr<PrimitiveObject> foundABCCurves(Alembic::AbcGeom::ICurvesSchema &mesh, int frameid, bool read_done) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = (int)std::round(start / time_per_cycle );

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    auto iSS = Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index);
    Alembic::AbcGeom::ICurvesSchema::Sample mesamp = mesh.getValue(iSS);
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
    read_velocity(prim, mesamp.getVelocities(), read_done);
    {
        auto &parr = prim->lines;
        auto numCurves = mesamp.getCurvesNumVertices()->size();
        std::size_t offset = 0;
        for (auto i = 0; i < numCurves; i++) {
            auto count = mesamp.getCurvesNumVertices()->operator[](i);
            for (auto j = 0; j < count-1; j++) {
                parr.push_back(vec2i(offset + j, offset + j + 1));
            }
            offset += count;
        }
    }
    ICompoundProperty arbattrs = mesh.getArbGeomParams();
    read_attributes(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = mesh.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);
    return prim;
}

void traverseABC(
    Alembic::AbcGeom::IObject &obj,
    ABCTree &tree,
    int frameid,
    bool read_done
) {
    {
        auto const &md = obj.getMetaData();
        if (!read_done) {
            log_debug("[alembic] meta data: [{}]", md.serialize());
        }
        tree.name = obj.getName();

        if (Alembic::AbcGeom::IPolyMesh::matches(md)) {
            if (!read_done) {
                log_debug("[alembic] found a mesh [{}]", obj.getName());
            }

            Alembic::AbcGeom::IPolyMesh meshy(obj);
            auto &mesh = meshy.getSchema();
            tree.prim = foundABCMesh(mesh, frameid, read_done);
            tree.prim->userData().set2("_abc_name", obj.getName());
        } else if (Alembic::AbcGeom::IXformSchema::matches(md)) {
            if (!read_done) {
                log_debug("[alembic] found a Xform [{}]", obj.getName());
            }
            Alembic::AbcGeom::IXform xfm(obj);
            auto &cam_sch = xfm.getSchema();
            tree.xform = foundABCXform(cam_sch, frameid);
        } else if (Alembic::AbcGeom::ICameraSchema::matches(md)) {
            if (!read_done) {
                log_debug("[alembic] found a Camera [{}]", obj.getName());
            }
            Alembic::AbcGeom::ICamera cam(obj);
            auto &cam_sch = cam.getSchema();
            tree.camera_info = foundABCCamera(cam_sch, frameid);
        } else if(Alembic::AbcGeom::IPointsSchema::matches(md)) {
            if (!read_done) {
                log_debug("[alembic] found points [{}]", obj.getName());
            }
            Alembic::AbcGeom::IPoints points(obj);
            auto &points_sch = points.getSchema();
            tree.prim = foundABCPoints(points_sch, frameid, read_done);
            tree.prim->userData().set2("_abc_name", obj.getName());
        } else if(Alembic::AbcGeom::ICurvesSchema::matches(md)) {
            if (!read_done) {
                log_debug("[alembic] found curves [{}]", obj.getName());
            }
            Alembic::AbcGeom::ICurves curves(obj);
            auto &curves_sch = curves.getSchema();
            tree.prim = foundABCCurves(curves_sch, frameid, read_done);
            tree.prim->userData().set2("_abc_name", obj.getName());
        }
    }

    size_t nch = obj.getNumChildren();
    if (!read_done) {
        log_debug("[alembic] found {} children", nch);
    }

    for (size_t i = 0; i < nch; i++) {
        auto const &name = obj.getChildHeader(i).getName();
        if (!read_done) {
            log_debug("[alembic] at {} name: [{}]", i, name);
        }

        Alembic::AbcGeom::IObject child(obj, name);

        auto childTree = std::make_shared<ABCTree>();
        traverseABC(child, *childTree, frameid, read_done);
        tree.children.push_back(std::move(childTree));
    }
}

Alembic::AbcGeom::IArchive readABC(std::string const &path) {
    std::string native_path = std::filesystem::u8path(path).string();
    std::string hdr;
    {
        char buf[5];
        std::memset(buf, 0, 5);
        auto fp = std::fopen(native_path.c_str(), "rb");
        if (!fp)
            throw Exception("[alembic] cannot open file for read: " + path);
        std::fread(buf, 4, 1, fp);
        std::fclose(fp);
        hdr = buf;
    }
    if (hdr == "\x89HDF") {
        log_info("[alembic] opening as HDF5 format");
        return {Alembic::AbcCoreHDF5::ReadArchive(), native_path};
    } else if (hdr == "Ogaw") {
        log_info("[alembic] opening as Ogawa format");
        return {Alembic::AbcCoreOgawa::ReadArchive(), native_path};
    } else {
        throw Exception("[alembic] unrecognized ABC header: [" + hdr + "]");
    }
}

struct ReadAlembic : INode {
    Alembic::Abc::v12::IArchive archive;
    std::string usedPath;
    bool read_done = false;
    virtual void apply() override {
        int frameid;
        if (has_input("frameid")) {
            frameid = get_input<NumericObject>("frameid")->get<int>();
        } else {
            frameid = getGlobalState()->frameid;
        }
        auto abctree = std::make_shared<ABCTree>();
        {
            auto path = get_input<StringObject>("path")->get();
            if (usedPath != path) {
                read_done = false;
            }
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
            usedPath = path;
        }
        set_output("abctree", std::move(abctree));
    }
};

ZENDEFNODE(ReadAlembic, {
    {
        {"readpath", "path"},
        {"frameid"},
    },
    {{"ABCTree", "abctree"}},
    {},
    {"alembic"},
});

} // namespace zeno
