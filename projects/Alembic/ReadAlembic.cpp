// https://github.com/alembic/alembic/blob/master/lib/Alembic/AbcGeom/Tests/PolyMeshTest.cpp
// WHY THE FKING ALEMBIC OFFICIAL GIVES NO DOC BUT ONLY "TESTS" FOR ME TO LEARN THEIR FKING LIB
#include <zeno/zeno.h>
#include <zeno/utils/logger.h>
#include <zeno/types/StringObject.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/types/PrimitiveTools.h>
#include <zeno/types/NumericObject.h>
#include <zeno/types/UserData.h>
#include <zeno/funcs/PrimitiveUtils.h>
#include <Alembic/AbcGeom/All.h>
#include <Alembic/AbcCoreAbstract/All.h>
#include <Alembic/AbcCoreOgawa/All.h>
#include <Alembic/AbcCoreHDF5/All.h>
#include <Alembic/Abc/ErrorHandler.h>
#include "ABCTree.h"
#include "zeno/types/DictObject.h"
#include "ABCCommon.h"
#include <cstring>
#include <cstdio>
#include <filesystem>
#include <zeno/utils/string.h>
#include <zeno/utils/scope_exit.h>
#include <numeric>

#ifdef ZENO_WITH_PYTHON3
    #include <Python.h>
#endif

using namespace Alembic::AbcGeom;

namespace zeno {
void TimeAndSamplesMap::add(TimeSamplingPtr iTime, size_t iNumSamples)
{

    if (iNumSamples == 0)
    {
        iNumSamples = 1;
    }

    for (size_t i = 0; i < mTimeSampling.size(); ++i)
    {
        if (mTimeSampling[i]->getTimeSamplingType() ==
            iTime->getTimeSamplingType())
        {
            chrono_t curLastTime =
                    mTimeSampling[i]->getSampleTime(mExpectedSamples[i]);

            chrono_t lastTime = iTime->getSampleTime(iNumSamples);
            if (lastTime < curLastTime)
            {
                lastTime = curLastTime;
            }

            if (mTimeSampling[i]->getSampleTime(0) > iTime->getSampleTime(0))
            {
                mTimeSampling[i] = iTime;
            }

            mExpectedSamples[i] = mTimeSampling[i]->getNearIndex(lastTime,
                                                                 std::numeric_limits< index_t >::max()).first;

            return;
        }
    }

    mTimeSampling.push_back(iTime);
    mExpectedSamples.push_back(iNumSamples);
}

TimeSamplingPtr TimeAndSamplesMap::get(TimeSamplingPtr iTime,
                                       std::size_t & oNumSamples) const
{
    for (size_t i = 0; i < mTimeSampling.size(); ++i)
    {
        if (mTimeSampling[i]->getTimeSamplingType() ==
            iTime->getTimeSamplingType())
        {
            oNumSamples = mExpectedSamples[i];
            return mTimeSampling[i];
        }
    }

    oNumSamples = 0;
    return TimeSamplingPtr();
}
static int clamp(int i, int _min, int _max) {
    if (i < _min) {
        return _min;
    } else if (i > _max) {
        return _max;
    } else {
        return i;
    }
}

static void set_time_info(UserData &ud, TimeSamplingType tst, float start, int sample_count) {
    float time_per_cycle = tst.getTimePerCycle();
    if (tst.isUniform()) {
        ud.set2("_abc_time_sampling_type", "Uniform");
    }
    else if (tst.isCyclic()) {
        ud.set2("_abc_time_sampling_type", "Cyclic");
    }
    else if (tst.isAcyclic()) {
        ud.set2("_abc_time_sampling_type", "Acyclic");
    }
    ud.set2("_abc_start_time", float(start));
    ud.set2("_abc_sample_count", sample_count);
    ud.set2("_abc_time_per_cycle", time_per_cycle);
    if (time_per_cycle > 0) {
        ud.set2("_abc_time_fps", 1.0f / time_per_cycle);
    }
    else {
        ud.set2("_abc_time_fps", 0.0f);
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
template<typename T>
void attr_from_data(std::shared_ptr<PrimitiveObject> prim, GeometryScope scope, std::string attr_name, std::vector<T> &data) {
    if (scope == GeometryScope::kUniformScope) {
        if (zeno::ends_with(attr_name, "_polys")) {
            attr_name = attr_name.substr(0, attr_name.size() - 6);
        }
        if (prim->polys.size() == data.size()) {
            auto &attr = prim->polys.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->polys.size() * 2 == data.size()) {
            auto &attr = prim->polys.add_attr<zeno::vec<2, T>>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = {data[2 * i], data[2 * i + 1]};
            }
        }
        else if (prim->polys.size() * 3 == data.size()) {
            auto &attr = prim->polys.add_attr<zeno::vec<3, T>>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        }
        else if (prim->polys.size() * 4 == data.size()) {
            auto &attr = prim->polys.add_attr<zeno::vec<4, T>>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = {data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]};
            }
        }
        else {
            log_warn("[alembic] can not load {} attr {}: {} in kUniformScope scope.", typeid(data[0]).name(), attr_name, data.size());
        }
    }
    else if (scope == GeometryScope::kFacevaryingScope) {
        if (zeno::ends_with(attr_name, "_loops")) {
            attr_name = attr_name.substr(0, attr_name.size() - 6);
        }
        if (prim->loops.size() == data.size()) {
            auto &attr = prim->loops.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->loops.size() * 2 == data.size()) {
            auto &attr = prim->loops.add_attr<zeno::vec<2, T>>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = {data[2 * i], data[2 * i + 1]};
            }
        }
        else if (prim->loops.size() * 3 == data.size()) {
            auto &attr = prim->loops.add_attr<zeno::vec<3, T>>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        }
        else if (prim->loops.size() * 4 == data.size()) {
            auto &attr = prim->loops.add_attr<zeno::vec<4, T>>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = {data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]};
            }
        }
        else {
            log_warn("[alembic] can not load {} attr {}: {} in kFacevaryingScope scope.", typeid(data[0]).name(), attr_name, data.size());
        }
    }
    else {
        if (prim->verts.size() == data.size()) {
            auto &attr = prim->verts.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->verts.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->verts.size() * 2 == data.size()) {
            auto &attr = prim->verts.add_attr<zeno::vec<2, T>>(attr_name);
            for (auto i = 0; i < prim->verts.size(); i++) {
                attr[i] = {data[2 * i], data[2 * i + 1]};
            }
        }
        else if (prim->verts.size() * 3 == data.size()) {
            auto &attr = prim->verts.add_attr<zeno::vec<3, T>>(attr_name);
            for (auto i = 0; i < prim->verts.size(); i++) {
                attr[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        }
        else if (prim->verts.size() * 4 == data.size()) {
            auto &attr = prim->verts.add_attr<zeno::vec<4, T>>(attr_name);
            for (auto i = 0; i < prim->verts.size(); i++) {
                attr[i] = {data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]};
            }
        }
        else if (prim->polys.size() == data.size()) {
            auto &attr = prim->polys.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->polys.size() * 2 == data.size()) {
            auto &attr = prim->polys.add_attr<zeno::vec<2, T>>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = {data[2 * i], data[2 * i + 1]};
            }
        }
        else if (prim->polys.size() * 3 == data.size()) {
            auto &attr = prim->polys.add_attr<zeno::vec<3, T>>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        }
        else if (prim->polys.size() * 4 == data.size()) {
            auto &attr = prim->polys.add_attr<zeno::vec<4, T>>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = {data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]};
            }
        }
        else if (prim->loops.size() == data.size()) {
            auto &attr = prim->loops.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->loops.size() * 2 == data.size()) {
            auto &attr = prim->loops.add_attr<zeno::vec<2, T>>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = {data[2 * i], data[2 * i + 1]};
            }
        }
        else if (prim->loops.size() * 3 == data.size()) {
            auto &attr = prim->loops.add_attr<zeno::vec<3, T>>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = {data[3 * i], data[3 * i + 1], data[3 * i + 2]};
            }
        }
        else if (prim->loops.size() * 4 == data.size()) {
            auto &attr = prim->loops.add_attr<zeno::vec<4, T>>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = {data[4 * i], data[4 * i + 1], data[4 * i + 2], data[4 * i + 3]};
            }
        }
        else {
            if (scope == GeometryScope::kVaryingScope) {
                log_warn("[alembic] can not load {} attr {}: {} in kVaryingScope scope.", typeid(data[0]).name(), attr_name, data.size());
            }
            else if (scope == GeometryScope::kVertexScope) {
                log_warn("[alembic] can not load {} attr {}: {} in kVertexScope scope.", typeid(data[0]).name(), attr_name, data.size());
            }
            else if (scope == GeometryScope::kUnknownScope) {
                log_warn("[alembic] can not load {} attr {}: {} in kUnknownScope scope.", typeid(data[0]).name(), attr_name, data.size());
            }
        }
    }
}
template<typename T>
void attr_from_data_vec(std::shared_ptr<PrimitiveObject> prim, GeometryScope scope, std::string attr_name, std::vector<T> &data) {
    if (scope == GeometryScope::kUniformScope) {
        if (prim->polys.size() == data.size()) {
            auto &attr = prim->polys.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = data[i];
            }
        }
        else {
            log_warn("[alembic] can not load {} attr {}: {} in kUniformScope scope.", typeid(data[0]).name(), attr_name, data.size());
        }
    }
    else if (scope == GeometryScope::kFacevaryingScope) {
        if (prim->loops.size() == data.size()) {
            auto &attr = prim->loops.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = data[i];
            }
        }
        else {
            log_warn("[alembic] can not load {} attr {}: {} in kFacevaryingScope scope.", typeid(data[0]).name(), attr_name, data.size());
        }
    }
    else {
        if (prim->verts.size() == data.size()) {
            auto &attr = prim->verts.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->verts.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->polys.size() == data.size()) {
            auto &attr = prim->polys.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->polys.size(); i++) {
                attr[i] = data[i];
            }
        }
        else if (prim->loops.size() == data.size()) {
            auto &attr = prim->loops.add_attr<T>(attr_name);
            for (auto i = 0; i < prim->loops.size(); i++) {
                attr[i] = data[i];
            }
        }
        else {
            if (scope == GeometryScope::kVaryingScope) {
                log_warn("[alembic] can not load {} attr {}: {} in kVaryingScope scope.", typeid(data[0]).name(), attr_name, data.size());
            }
            else if (scope == GeometryScope::kVertexScope) {
                log_warn("[alembic] can not load {} attr {}: {} in kVertexScope scope.", typeid(data[0]).name(), attr_name, data.size());
            }
            else if (scope == GeometryScope::kUnknownScope) {
                log_warn("[alembic] can not load {} attr {}: {} in kUnknownScope scope.", typeid(data[0]).name(), attr_name, data.size());
            }
        }
    }
}
static void read_attributes2(std::shared_ptr<PrimitiveObject> prim, ICompoundProperty arbattrs, const ISampleSelector &iSS, bool read_done) {
    if (!arbattrs) {
        return;
    }
    size_t numProps = arbattrs.getNumProperties();
    for (auto i = 0; i < numProps; i++) {
        PropertyHeader p = arbattrs.getPropertyHeader(i);
        if (IFloatGeomParam::matches(p)) {
            IFloatGeomParam param(arbattrs, p.getName());

            IFloatGeomParam::Sample samp = param.getExpandedValue(iSS);
            std::vector<float> data;
            data.resize(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                data[i] = samp.getVals()->get()[i];
            }
            if (!read_done) {
                log_info("[alembic] float attr {}, len {}.", p.getName(), data.size());
            }
            attr_from_data(prim, samp.getScope(), p.getName(), data);
        }
        else if (IInt32GeomParam::matches(p)) {
            IInt32GeomParam param(arbattrs, p.getName());

            IInt32GeomParam::Sample samp = param.getExpandedValue(iSS);
            std::vector<int> data;
            data.resize(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                data[i] = samp.getVals()->get()[i];
            }
            if (!read_done) {
                log_info("[alembic] int attr {}, len {}.", p.getName(), data.size());
            }
            attr_from_data(prim, samp.getScope(), p.getName(), data);
        }
        else if (IV3fGeomParam::matches(p)) {
            IV3fGeomParam param(arbattrs, p.getName());

            IV3fGeomParam::Sample samp = param.getExpandedValue(iSS);
            std::vector<vec3f> data;
            data.resize(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                auto v = samp.getVals()->get()[i];
                data[i] = {v[0], v[1], v[2]};
            }
            if (!read_done) {
                log_info("[alembic] V3f attr {}, len {}.", p.getName(), data.size());
            }
            attr_from_data_vec(prim, samp.getScope(), p.getName(), data);
        }
        else if (IN3fGeomParam::matches(p)) {
            IN3fGeomParam param(arbattrs, p.getName());

            IN3fGeomParam::Sample samp = param.getExpandedValue(iSS);
            std::vector<vec3f> data;
            data.resize(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                auto v = samp.getVals()->get()[i];
                data[i] = {v[0], v[1], v[2]};
            }
            if (!read_done) {
                log_info("[alembic] N3f attr {}, len {}.", p.getName(), data.size());
            }
            attr_from_data_vec(prim, samp.getScope(), p.getName(), data);
        }
        else if (IC3fGeomParam::matches(p)) {
            IC3fGeomParam param(arbattrs, p.getName());

            IC3fGeomParam::Sample samp = param.getExpandedValue(iSS);
            std::vector<vec3f> data;
            data.resize(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                auto v = samp.getVals()->get()[i];
                data[i] = {v[0], v[1], v[2]};
            }
            if (!read_done) {
                log_info("[alembic] C3f attr {}, len {}.", p.getName(), data.size());
            }
            attr_from_data_vec(prim, samp.getScope(), p.getName(), data);
        }
        else if (IC4fGeomParam::matches(p)) {
            IC4fGeomParam param(arbattrs, p.getName());

            IC4fGeomParam::Sample samp = param.getExpandedValue(iSS);
            std::vector<vec4f> data;
            data.resize(samp.getVals()->size());
            std::vector<vec3f> data_xyz(samp.getVals()->size());
            std::vector<float> data_w(samp.getVals()->size());
            for (auto i = 0; i < samp.getVals()->size(); i++) {
                auto v = samp.getVals()->get()[i];
                data[i] = {v[0], v[1], v[2], v[3]};
                data_xyz[i] = {v[0], v[1], v[2]};
                data_w[i] = v[3];
            }
            if (!read_done) {
                log_info("[alembic] C4f attr {}, len {}.", p.getName(), data.size());
            }
            attr_from_data_vec(prim, samp.getScope(), p.getName(), data);
            attr_from_data_vec(prim, samp.getScope(), p.getName() + "_rgb", data_xyz);
            attr_from_data_vec(prim, samp.getScope(), p.getName() + "_a", data_w);
        }
        else {
            log_info("[alembic] unknown attr {}.", p.getName());
            zeno::log_info("getExtent {} ", p.getDataType().getExtent());
            zeno::log_info("getNumBytes {} ", p.getDataType().getNumBytes());
            zeno::log_info("getPod {} ", p.getDataType().getPod());
        }
    }
    {
        if (prim->loops.attr_keys<AttrAcceptAll>().size() == 0) {
            return;
        }
        if (prim->loops.attr_keys<AttrAcceptAll>().size() == 1 && prim->loops.has_attr("uvs")) {
            return;
        }
        if (!prim->loops.has_attr("uvs")) {
            prim->loops.add_attr<int>("uvs");
            prim->uvs.emplace_back();
        }
        {
            std::vector<vec2f> uvs(prim->loops.size());
            auto &uv_index = prim->loops.attr<int>("uvs");
            for (auto i = 0; i < prim->loops.size(); i++) {
                uvs[i] = prim->uvs[uv_index[i]];
            }
            prim->uvs.values = uvs;
            std::iota(uv_index.begin(), uv_index.end(), 0);
            prim->loops.foreach_attr<AttrAcceptAll>([&] (auto const &key, auto &arr) {
                if (key == "uvs") {
                    return;
                }
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = prim->uvs.add_attr<T>(key);
                std::copy(arr.begin(), arr.end(), attr.begin());
            });
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
        else if (IBoolProperty::matches(p)) {
            IBoolProperty param(arbattrs, p.getName());

            auto value = param.getValue(iSS);
            prim->userData().set2(p.getName(), int(value));
        }
        else if (IInt16Property::matches(p)) {
            IInt16Property param(arbattrs, p.getName());

            auto value = param.getValue(iSS);
            prim->userData().set2(p.getName(), int(value));
        }
        else {
            if (!read_done) {
                log_warn("[alembic] can not load user data {}..", p.getName());
            }
        }
    }
}

static ObjectVisibility read_visible_attr(ICompoundProperty arbattrs, const ISampleSelector &iSS) {
    if (!arbattrs) {
        return ObjectVisibility::kVisibilityDeferred;
    }
    size_t numProps = arbattrs.getNumProperties();
    for (auto i = 0; i < numProps; i++) {
        PropertyHeader p = arbattrs.getPropertyHeader(i);
        if (p.getName() != "visible") {
            continue;
        }
        if (ICharProperty::matches(p)) {
            ICharProperty param(arbattrs, p.getName());

            auto value = param.getValue(iSS);
            if (value == 0) {
                return ObjectVisibility::kVisibilityHidden;
            }
            else if (value == 1) {
                return ObjectVisibility::kVisibilityVisible;
            }
            else {
                return ObjectVisibility::kVisibilityDeferred;
            }
        }
    }
    return ObjectVisibility::kVisibilityDeferred;
}

static std::shared_ptr<PrimitiveObject> foundABCMesh(
        Alembic::AbcGeom::IPolyMeshSchema &mesh
        , int frameid
        , bool read_done
        , bool read_face_set
        , bool outOfRangeAsEmpty
        , std::string abc_name
) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = std::lround(start / time_per_cycle );
    set_time_info(prim->userData(), time->getTimeSamplingType(), start, int(mesh.getNumSamples()));

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    if (outOfRangeAsEmpty && frameid - start_frame != sample_index) {
        return prim;
    }
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

    bool is_point = true;

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
            if (cnt != 1) {
                is_point = false;
            }
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
    read_attributes2(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = mesh.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);

    if (is_point) {
        prim->loops.clear();
        prim->polys.clear();
        return prim;
    }

    if (read_face_set) {
        auto &faceset = prim->polys.add_attr<int>("faceset");
        std::fill(faceset.begin(), faceset.end(), -1);
        auto &ud = prim->userData();
        std::vector<std::string> faceSetNames;
        mesh.getFaceSetNames(faceSetNames);
        for (auto i = 0; i < faceSetNames.size(); i++) {
            auto n = faceSetNames[i];
            IFaceSet faceSet = mesh.getFaceSet(n);
            IFaceSetSchema::Sample faceSetSample = faceSet.getSchema().getValue();
            size_t s = faceSetSample.getFaces()->size();
            for (auto j = 0; j < s; j++) {
                int f = faceSetSample.getFaces()->get()[j];
                faceset[f] = i;
            }
        }
        bool found_unbind_faces = false;
        int next_faceset_index = faceSetNames.size();
        for (auto i = 0; i < faceset.size(); i++) {
            if (faceset[i] == -1) {
                found_unbind_faces = true;
                faceset[i] = next_faceset_index;
            }
        }
        if (found_unbind_faces) {
            faceSetNames.push_back(abc_name);
        }
        for (auto i = 0; i < faceSetNames.size(); i++) {
            auto n = faceSetNames[i];
            ud.set2(zeno::format("faceset_{}", i), n);
        }
        ud.set2("faceset_count", int(faceSetNames.size()));
    }

    return prim;
}

static std::shared_ptr<PrimitiveObject> foundABCSubd(Alembic::AbcGeom::ISubDSchema &subd, int frameid, bool read_done, bool read_face_set, bool outOfRangeAsEmpty) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = subd.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = std::lround(start / time_per_cycle );
    set_time_info(prim->userData(), time->getTimeSamplingType(), start, int(subd.getNumSamples()));

    int sample_index = clamp(frameid - start_frame, 0, (int)subd.getNumSamples() - 1);
    if (outOfRangeAsEmpty && frameid - start_frame != sample_index) {
        return prim;
    }
    ISampleSelector iSS = Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index);
    Alembic::AbcGeom::ISubDSchema::Sample mesamp = subd.getValue(iSS);

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
    if (auto uv = subd.getUVsParam()) {
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
            // log_warn("[alembic] Not found uv, auto fill zero.");
        }
        prim->uvs.resize(1);
        prim->uvs[0] = zeno::vec2f(0, 0);
        prim->loops.add_attr<int>("uvs");
        for (auto i = 0; i < prim->loops.size(); i++) {
            prim->loops.attr<int>("uvs")[i] = 0;
        }
    }
    ICompoundProperty arbattrs = subd.getArbGeomParams();
    read_attributes2(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = subd.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);

    if (read_face_set) {
        auto &faceset = prim->polys.add_attr<int>("faceset");
        std::fill(faceset.begin(), faceset.end(), -1);
        auto &ud = prim->userData();
        std::vector<std::string> faceSetNames;
        subd.getFaceSetNames(faceSetNames);
        ud.set2("faceset_count", int(faceSetNames.size()));
        for (auto i = 0; i < faceSetNames.size(); i++) {
            auto n = faceSetNames[i];
            ud.set2(zeno::format("faceset_{}", i), n);
            IFaceSet faceSet = subd.getFaceSet(n);
            IFaceSetSchema::Sample faceSetSample = faceSet.getSchema().getValue();
            size_t s = faceSetSample.getFaces()->size();
            for (auto j = 0; j < s; j++) {
                int f = faceSetSample.getFaces()->get()[j];
                faceset[f] = i;
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
    int start_frame = std::lround(start / time_per_cycle );
    int sample_index = clamp(frameid - start_frame, 0, (int)cam.getNumSamples() - 1);

    auto samp = cam.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
    cam_info.focal_length = samp.getFocalLength();
    cam_info._near = samp.getNearClippingPlane();
    cam_info._far = samp.getFarClippingPlane();
    cam_info.horizontalAperture = samp.getHorizontalAperture() * 10;
    cam_info.verticalAperture = samp.getVerticalAperture() * 10;
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
    int start_frame = std::lround(start / time_per_cycle );
    int sample_index = clamp(frameid - start_frame, 0, (int)xfm.getNumSamples() - 1);

    auto samp = xfm.getValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
    return samp.getMatrix();
}

static std::shared_ptr<PrimitiveObject> foundABCPoints(Alembic::AbcGeom::IPointsSchema &mesh, int frameid, bool read_done, bool outOfRangeAsEmpty) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = std::lround(start / time_per_cycle );
    set_time_info(prim->userData(), time->getTimeSamplingType(), start, int(mesh.getNumSamples()));

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    if (outOfRangeAsEmpty && frameid - start_frame != sample_index) {
        return prim;
    }
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

    {
        auto &ids = prim->verts.add_attr<int>("id");
        auto count = mesamp.getIds()->size();
        for (auto i = 0; i < count; i++) {
            ids[i] = mesamp.getIds()->operator[](i);
        }
    }
    read_velocity(prim, mesamp.getVelocities(), read_done);
    ICompoundProperty arbattrs = mesh.getArbGeomParams();
    read_attributes2(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = mesh.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);
    return prim;
}

static std::shared_ptr<PrimitiveObject> foundABCCurves(Alembic::AbcGeom::ICurvesSchema &mesh, int frameid, bool read_done, bool outOfRangeAsEmpty) {
    auto prim = std::make_shared<PrimitiveObject>();

    std::shared_ptr<Alembic::AbcCoreAbstract::v12::TimeSampling> time = mesh.getTimeSampling();
    float time_per_cycle =  time->getTimeSamplingType().getTimePerCycle();
    double start = time->getStoredTimes().front();
    int start_frame = std::lround(start / time_per_cycle );
    set_time_info(prim->userData(), time->getTimeSamplingType(), start, int(mesh.getNumSamples()));

    int sample_index = clamp(frameid - start_frame, 0, (int)mesh.getNumSamples() - 1);
    if (outOfRangeAsEmpty && frameid - start_frame != sample_index) {
        return prim;
    }
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
    if (auto width = mesh.getWidthsParam()) {
        auto widthsamp =
            width.getIndexedValue(Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index));
        int index_size = (int)widthsamp.getIndices()->size();
        if (prim->verts.size() == index_size) {
            auto &width_attr = prim->add_attr<float>("width");
            for (auto i = 0; i < prim->verts.size(); i++) {
                auto index = widthsamp.getIndices()->operator[](i);
                auto value = widthsamp.getVals()->operator[](index);
                width_attr[i] = value;
            }
        }
    }
    ICompoundProperty arbattrs = mesh.getArbGeomParams();
    read_attributes2(prim, arbattrs, iSS, read_done);
    ICompoundProperty usrData = mesh.getUserProperties();
    read_user_data(prim, usrData, iSS, read_done);
    return prim;
}

void traverseABC(
    Alembic::AbcGeom::IObject &obj,
    ABCTree &tree,
    int frameid,
    bool read_done,
    bool read_face_set,
    std::string path,
    const TimeAndSamplesMap & iTimeMap,
    ObjectVisibility parent_visible,
    bool skipInvisibleObject,
    bool outOfRangeAsEmpty
) {
    {
        auto const &md = obj.getMetaData();
        if (!read_done) {
            log_debug("[alembic] meta data: [{}]", md.serialize());
        }
        tree.name = obj.getName();
        path = zeno::format("{}/{}", path, tree.name);
        auto visible_prop = obj.getProperties().getPropertyHeader("visible");
        if (visible_prop) {
            size_t totalSamples = 0;
            TimeSamplingPtr timePtr =
                    iTimeMap.get(visible_prop->getTimeSampling(), totalSamples);
            float time_per_cycle = visible_prop->getTimeSampling()->getTimeSamplingType().getTimePerCycle();
            double start = visible_prop->getTimeSampling()->getStoredTimes().front();
            int start_frame = std::lround(start / time_per_cycle );

            int sample_index = clamp(frameid - start_frame, 0, (int)totalSamples - 1);
            ISampleSelector iSS = Alembic::Abc::v12::ISampleSelector((Alembic::AbcCoreAbstract::index_t)sample_index);
            auto visible = read_visible_attr(obj.getProperties(), iSS);
            if (visible != -1) {
                tree.visible = visible;
            }
            else {
                tree.visible = parent_visible;
            }
        }
        else {
            tree.visible = parent_visible;
        }
        if (!(tree.visible == ObjectVisibility::kVisibilityHidden && skipInvisibleObject)) {
            if (Alembic::AbcGeom::IPolyMesh::matches(md)) {
                if (!read_done) {
                    log_debug("[alembic] found a mesh [{}]", obj.getName());
                }

                Alembic::AbcGeom::IPolyMesh meshy(obj);
                auto &mesh = meshy.getSchema();
                tree.prim = foundABCMesh(mesh, frameid, read_done, read_face_set, outOfRangeAsEmpty, obj.getName());
                tree.prim->userData().set2("_abc_name", obj.getName());
                prim_set_abcpath(tree.prim.get(), path);
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
                tree.prim = foundABCPoints(points_sch, frameid, read_done, outOfRangeAsEmpty);
                tree.prim->userData().set2("_abc_name", obj.getName());
                prim_set_abcpath(tree.prim.get(), path);
                tree.prim->userData().set2("faceset_count", 0);
            } else if(Alembic::AbcGeom::ICurvesSchema::matches(md)) {
                if (!read_done) {
                    log_debug("[alembic] found curves [{}]", obj.getName());
                }
                Alembic::AbcGeom::ICurves curves(obj);
                auto &curves_sch = curves.getSchema();
                tree.prim = foundABCCurves(curves_sch, frameid, read_done, outOfRangeAsEmpty);
                tree.prim->userData().set2("_abc_name", obj.getName());
                prim_set_abcpath(tree.prim.get(), path);
                tree.prim->userData().set2("faceset_count", 0);
            } else if (Alembic::AbcGeom::ISubDSchema::matches(md)) {
                if (!read_done) {
                    log_debug("[alembic] found SubD [{}]", obj.getName());
                }
                Alembic::AbcGeom::ISubD subd(obj);
                auto &subd_sch = subd.getSchema();
                tree.prim = foundABCSubd(subd_sch, frameid, read_done, read_face_set, outOfRangeAsEmpty);
                tree.prim->userData().set2("_abc_name", obj.getName());
                prim_set_abcpath(tree.prim.get(), path);
            }
            if (tree.prim) {
                tree.prim->userData().set2("vis", tree.visible);
                if (tree.visible == 0) {
                    for (auto i = 0; i < tree.prim->verts.size(); i++) {
                        tree.prim->verts[i] = {};
                    }
                }
            }
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
        traverseABC(child, *childTree, frameid, read_done, read_face_set, path, iTimeMap, tree.visible, skipInvisibleObject, outOfRangeAsEmpty);
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
            frameid = std::lround(get_input<NumericObject>("frameid")->get<float>());
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
            bool read_face_set = get_input2<bool>("read_face_set");
            bool outOfRangeAsEmpty = get_input2<bool>("outOfRangeAsEmpty");
            bool skipInvisibleObject = get_input2<bool>("skipInvisibleObject");
            Alembic::Util::uint32_t numSamplings = archive.getNumTimeSamplings();
            TimeAndSamplesMap timeMap;
            for (Alembic::Util::uint32_t s = 0; s < numSamplings; ++s)             {
                timeMap.add(archive.getTimeSampling(s),
                            archive.getMaxNumSamplesForTimeSamplingIndex(s));
            }

            traverseABC(obj, *abctree, frameid, read_done, read_face_set, "", timeMap, ObjectVisibility::kVisibilityDeferred,
                        skipInvisibleObject, outOfRangeAsEmpty);
            read_done = true;
            usedPath = path;
        }
        {
            auto namelist = std::make_shared<zeno::ListObject>();
            abctree->visitPrims([&] (auto const &p) {
                auto &ud = p->userData();
                auto _abc_path = ud.template get2<std::string>("abcpath_0", "");
                namelist->arr.push_back(std::make_shared<StringObject>(_abc_path));
            });
            auto &ud = abctree->userData();
            ud.set2("prim_count", int(namelist->arr.size()));
            for (auto i = 0; i < namelist->arr.size(); i++) {
                auto n = namelist->arr[i];
                ud.set2(zeno::format("path_{:04}", i), n);
            }
            set_output("namelist", namelist);
        }
        set_output("abctree", std::move(abctree));
    }
};

ZENDEFNODE(ReadAlembic, {
    {
        {"readpath", "path"},
        {"bool", "read_face_set", "1"},
        {"bool", "outOfRangeAsEmpty", "0"},
        {"bool", "skipInvisibleObject", "1"},
        {"frameid"},
    },
    {
        {"ABCTree", "abctree"},
        "namelist",
    },
    {},
    {"alembic"},
});

std::shared_ptr<ListObject> abc_split_by_name(std::shared_ptr<PrimitiveObject> prim, bool add_when_none) {
    auto list = std::make_shared<ListObject>();
    if (prim->verts.size() == 0) {
        return list;
    }
    int faceset_count = prim->userData().get2<int>("faceset_count");
    if (add_when_none && faceset_count == 0) {
        auto name = prim->userData().get2<std::string>("_abc_name");
        prim_set_faceset(prim.get(), name);
        faceset_count = 1;
    }
    std::map<int, std::vector<int>> faceset_map;
    for (auto f = 0; f < faceset_count; f++) {
        faceset_map[f] = {};
    }
    if (prim->polys.size()) {
        auto &faceset = prim->polys.add_attr<int>("faceset");
        for (auto j = 0; j < faceset.size(); j++) {
            auto f = faceset[j];
            faceset_map[f].push_back(j);
        }
        for (auto f = 0; f < faceset_count; f++) {
            auto name = prim->userData().get2<std::string>(zeno::format("faceset_{}", f));
            auto new_prim = std::dynamic_pointer_cast<PrimitiveObject>(prim->clone());
            new_prim->polys.resize(faceset_map[f].size());
            for (auto i = 0; i < faceset_map[f].size(); i++) {
                new_prim->polys[i] = prim->polys[faceset_map[f][i]];
            }
            new_prim->polys.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = prim->polys.attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    arr[i] = attr[faceset_map[f][i]];
                }
            });
            for (auto j = 0; j < faceset_count; j++) {
                new_prim->userData().del(zeno::format("faceset_{}", j));
            }
            prim_set_faceset(new_prim.get(), name);
            list->arr.push_back(new_prim);
        }
    }
    else if (prim->tris.size()) {
        auto &faceset = prim->tris.add_attr<int>("faceset");
        for (auto j = 0; j < faceset.size(); j++) {
            auto f = faceset[j];
            faceset_map[f].push_back(j);
        }
        for (auto f = 0; f < faceset_count; f++) {
            auto name = prim->userData().get2<std::string>(zeno::format("faceset_{}", f));
            auto new_prim = std::dynamic_pointer_cast<PrimitiveObject>(prim->clone());
            new_prim->tris.resize(faceset_map[f].size());
            for (auto i = 0; i < faceset_map[f].size(); i++) {
                new_prim->tris[i] = prim->tris[faceset_map[f][i]];
            }
            new_prim->tris.foreach_attr<AttrAcceptAll>([&](auto const &key, auto &arr) {
                using T = std::decay_t<decltype(arr[0])>;
                auto &attr = prim->tris.attr<T>(key);
                for (auto i = 0; i < arr.size(); i++) {
                    arr[i] = attr[faceset_map[f][i]];
                }
            });
            for (auto j = 0; j < faceset_count; j++) {
                new_prim->userData().del(zeno::format("faceset_{}", j));
            }
            prim_set_faceset(new_prim.get(), name);
            list->arr.push_back(new_prim);
        }
    }
    return list;
}
struct AlembicSplitByName: INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        int faceset_count = prim->userData().get2<int>("faceset_count");
        {
            auto namelist = std::make_shared<zeno::ListObject>();
            for (auto f = 0; f < faceset_count; f++) {
                auto name = prim->userData().get2<std::string>(zeno::format("faceset_{}", f));
                namelist->arr.push_back(std::make_shared<StringObject>(name));
            }
            set_output("namelist", namelist);
        }

        auto dict = std::make_shared<zeno::DictObject>();
        auto list = abc_split_by_name(prim, false);
        for (auto& prim: list->get<PrimitiveObject>()) {
            auto name = prim->userData().get2<std::string>("faceset_0");
            if (get_input2<bool>("killDeadVerts")) {
                primKillDeadVerts(prim.get());
            }
            dict->lut[name] = std::move(prim);
        }
        set_output("dict", dict);
    }
};

ZENDEFNODE(AlembicSplitByName, {
    {
        {"prim"},
        {"bool", "killDeadVerts", "1"},
    },
    {
        {"DictObject", "dict"},
        {"ListObject", "namelist"},
    },
    {},
    {"alembic"},
});

struct CopyPosAndNrmByIndex: INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto prims = get_input<ListObject>("list")->get<PrimitiveObject>();
        for (auto p: prims) {
            size_t size = p->size();
            auto index = p->attr<int>("index");
            for (auto i = 0; i < size; i++) {
                prim->verts[index[i]] = p->verts[i];
            }
            if (prim->verts.attr_is<vec3f>("nrm")) {
                auto &nrm = prim->verts.attr<vec3f>("nrm");
                auto &nrm_sub = p->verts.attr<vec3f>("nrm");
                for (auto i = 0; i < size; i++) {
                    nrm[index[i]] = nrm_sub[i];
                }
            }
        }

        set_output("out", prim);
    }
};

ZENDEFNODE(CopyPosAndNrmByIndex, {
    {
        {"prim"},
        {"list", "list"},
    },
    {
        {"out"},
    },
    {},
    {"alembic"},
});

struct PrimsFilterInUserdata: INode {
    void apply() override {
        auto prims = get_input<ListObject>("list")->get<PrimitiveObject>();
        auto filter_str = get_input2<std::string>("filters");
        std::vector<std::string> filters = zeno::split_str(filter_str, {' ', '\n'});
        std::vector<std::string> filters_;
        auto out_list = std::make_shared<ListObject>();

        for (auto &s: filters) {
            if (s.length() > 0) {
                filters_.push_back(s);
            }
        }

        auto name = get_input2<std::string>("name");
        auto contain = get_input2<bool>("contain");
        auto fuzzy = get_input2<bool>("fuzzy");
        for (auto p: prims) {
            auto &ud = p->userData();
            bool this_contain = false;
            if (ud.has<std::string>(name)) {
                if (fuzzy) {
                    for (auto & filter: filters_) {
                        if (ud.get2<std::string>(name).find(filter) != std::string::npos) {
                            this_contain = this_contain || true;
                        }
                    }
                }
                else {
                    this_contain = std::count(filters_.begin(), filters_.end(), ud.get2<std::string>(name)) > 0;
                }
            }
            else if (ud.has<int>(name)) {
                this_contain = std::count(filters_.begin(), filters_.end(), std::to_string(ud.get2<int>(name))) > 0;
            }
            else if (ud.has<float>(name)) {
                this_contain = std::count(filters_.begin(), filters_.end(), std::to_string(ud.get2<float>(name))) > 0;
            }
            bool insert = (contain && this_contain) || (!contain && !this_contain);
            if (insert) {
                out_list->arr.push_back(p);
            }
        }
        set_output("out", out_list);
    }
};

ZENDEFNODE(PrimsFilterInUserdata, {
    {
        {"list", "list"},
        {"string", "name"},
        {"multiline_string", "filters"},
        {"bool", "contain", "1"},
        {"bool", "fuzzy", "0"},
    },
    {
        {"list", "out"},
    },
    {},
    {"alembic"},
});

struct PrimsFilterInUserdataMultiTags: INode {
    void apply() override {
        auto prims = get_input<ListObject>("list")->get<PrimitiveObject>();
        auto filter_str = get_input2<std::string>("filters");
        std::vector<std::string> filters = zeno::split_str(filter_str, {' ', '\n'});
        std::vector<std::string> filters_;
        auto output = std::make_shared<DictObject>();

        for (auto &s: filters) {
            if (s.length() > 0) {
                filters_.push_back(s);
            }
        }

        auto name = get_input2<std::string>("name");
        auto fuzzy = get_input2<bool>("fuzzy");
        for (auto p: prims) {
            auto &ud = p->userData();
            if (ud.has<std::string>(name)) {
                if (fuzzy) {
                    for (auto & filter: filters_) {
                        if (ud.get2<std::string>(name).find(filter) != std::string::npos) {
                            if (!output->lut.count(filter)) {
                                output->lut[filter] = std::make_shared<ListObject>();
                            }
                            auto ptr = std::dynamic_pointer_cast<ListObject>(output->lut[filter]);
                            ptr->arr.push_back(p);
                        }
                    }
                }
                else {
                    auto value = ud.get2<std::string>(name);
                    if (std::count(filters_.begin(), filters_.end(), value)) {
                        if (!output->lut.count(value)) {
                            output->lut[value] = std::make_shared<ListObject>();
                        }
                        auto ptr = std::dynamic_pointer_cast<ListObject>(output->lut[value]);
                        ptr->arr.push_back(p);
                    }
                }
            }
            else if (ud.has<int>(name)) {
                auto value = std::to_string(ud.get2<int>(name));
                if (std::count(filters_.begin(), filters_.end(), value)) {
                    if (!output->lut.count(value)) {
                        output->lut[value] = std::make_shared<ListObject>();
                    }
                    auto ptr = std::dynamic_pointer_cast<ListObject>(output->lut[value]);
                    ptr->arr.push_back(p);
                }
            }
            else if (ud.has<float>(name)) {
                auto value = std::to_string(ud.get2<float>(name));
                if (std::count(filters_.begin(), filters_.end(), value)) {
                    if (!output->lut.count(value)) {
                        output->lut[value] = std::make_shared<ListObject>();
                    }
                    auto ptr = std::dynamic_pointer_cast<ListObject>(output->lut[value]);
                    ptr->arr.push_back(p);
                }
            }
        }
        set_output("out", output);
    }
};

ZENDEFNODE(PrimsFilterInUserdataMultiTags, {
    {
        {"list", "list"},
        {"string", "name"},
        {"multiline_string", "filters"},
        {"bool", "fuzzy", "0"},
    },
    {
        {"DictObject", "out"},
    },
    {},
    {"alembic"},
});

#ifdef ZENO_WITH_PYTHON3
static PyObject * pycheck(PyObject *pResult) {
    if (pResult == nullptr) {
        PyErr_Print();
        throw zeno::makeError("python err");
    }
    return pResult;
}

static void pycheck(int result) {
    if (result != 0) {
        PyErr_Print();
        throw zeno::makeError("python err");
    }
}
struct PrimsFilterInUserdataPython: INode {
    void apply() override {
        auto prims = get_input<ListObject>("list")->get<PrimitiveObject>();
        auto py_code = get_input2<std::string>("py_code");
        Py_Initialize();
        zeno::scope_exit init_defer([=]{ Py_Finalize(); });
        PyRun_SimpleString("import sys; sys.stderr = sys.stdout");

        auto out_list = std::make_shared<ListObject>();
        for (auto p: prims) {
            PyObject* userGlobals = PyDict_New();
            zeno::scope_exit userGlobals_defer([=]{ Py_DECREF(userGlobals); });

            PyObject* innerDict = PyDict_New();
            zeno::scope_exit innerDict_defer([=]{ Py_DECREF(innerDict); });

            auto &ud = p->userData();
            for (auto i = ud.begin(); i != ud.end(); i++) {
                auto key = i->first;
                if (ud.has<std::string>(key)) {
                    auto value = ud.get2<std::string>(key);
                    PyObject* pyInnerValue = PyUnicode_DecodeUTF8(key.c_str(), key.size(), "strict");
                    pycheck(PyDict_SetItemString(innerDict, key.c_str(), pyInnerValue));
                }
                else if (ud.has<float>(key)) {
                    auto value = ud.get2<float>(key);
                    PyObject* pyInnerValue = PyFloat_FromDouble(value);
                    pycheck(PyDict_SetItemString(innerDict, key.c_str(), pyInnerValue));
                }
                else if (ud.has<int>(key)) {
                    auto value = ud.get2<int>(key);
                    PyObject* pyInnerValue = PyLong_FromLong(value);
                    pycheck(PyDict_SetItemString(innerDict, key.c_str(), pyInnerValue));
                }
            }

            PyDict_SetItemString(userGlobals, "ud", innerDict);

            PyObject* pResult = pycheck(PyRun_String(py_code.c_str(), Py_file_input, userGlobals, nullptr));
            zeno::scope_exit pResult_defer([=]{ Py_DECREF(pResult); });
            PyObject* pValue = pycheck(PyRun_String("result", Py_eval_input, userGlobals, nullptr));
            zeno::scope_exit pValue_defer([=]{ Py_DECREF(pValue); });
            int need_insert = PyLong_AsLong(pValue);

            if (need_insert > 0) {
                out_list->arr.push_back(p);
            }
        }
        set_output("out", out_list);
    }
};

ZENDEFNODE(PrimsFilterInUserdataPython, {
    {
        {"list", "list"},
        {"multiline_string", "py_code", "result = len(ud['label']) > 2"},
    },
    {
        {"out"},
    },
    {},
    {"alembic"},
});

#endif
struct SetFaceset: INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto faceset_name = get_input2<std::string>("facesetName");
        prim_set_faceset(prim.get(), faceset_name);

        set_output("out", prim);
    }
};

ZENDEFNODE(SetFaceset, {
    {
        "prim",
        {"string", "facesetName", "defFS"},
    },
    {
        {"out"},
    },
    {},
    {"alembic"},
});

struct SetABCPath: INode {
    void apply() override {
        auto prim = get_input<PrimitiveObject>("prim");
        auto abcpathName = get_input2<std::string>("abcpathName");
        prim_set_abcpath(prim.get(), abcpathName);

        set_output("out", prim);
    }
};

ZENDEFNODE(SetABCPath, {
    {
        "prim",
        {"string", "abcpathName", "/ABC/your_path"},
    },
    {
        {"out"},
    },
    {},
    {"alembic"},
});


} // namespace zeno
