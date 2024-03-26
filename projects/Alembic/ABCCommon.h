#ifndef ZENO_ABCCOMMON_H
#define ZENO_ABCCOMMON_H

#include "ABCTree.h"
#include "Alembic/Abc/IObject.h"
#include "zeno/ListObject.h"

namespace zeno {
class TimeAndSamplesMap {
public:
    TimeAndSamplesMap() {m_isVerbose = false;};

    void add(Alembic::AbcCoreAbstract::TimeSamplingPtr iTime,
             std::size_t iNumSamples);

    Alembic::AbcCoreAbstract::TimeSamplingPtr get(
            Alembic::AbcCoreAbstract::TimeSamplingPtr iTime,
            std::size_t & oNumSamples) const;

    void setVerbose(bool isVerbose){m_isVerbose = isVerbose;};
    bool isVerbose() const {return m_isVerbose;};

private:
    std::vector< Alembic::AbcCoreAbstract::TimeSamplingPtr > mTimeSampling;
    std::vector< std::size_t > mExpectedSamples;
    bool m_isVerbose;
};

extern void traverseABC(
    Alembic::AbcGeom::IObject &obj,
    ABCTree &tree,
    int frameid,
    bool read_done,
    bool read_face_set,
    std::string path,
    const TimeAndSamplesMap & iTimeMap,
    ObjectVisibility parent_visible,
    bool outOfRangeAsEmpty
);

extern Alembic::AbcGeom::IArchive readABC(std::string const &path);

extern std::shared_ptr<zeno::ListObject> get_xformed_prims(std::shared_ptr<zeno::ABCTree> abctree);

extern std::shared_ptr<PrimitiveObject> get_alembic_prim(std::shared_ptr<zeno::ABCTree> abctree, int index);

void writeObjFile(
    const std::shared_ptr<zeno::PrimitiveObject>& primitive,
    const char *path,
    int32_t frameNum = 1,
    const std::pair<zeno::vec3f, zeno::vec3f>& bbox = std::make_pair(vec3f{}, vec3f{})
);

bool SaveEXR(const float* rgb, int width, int height, const char* outfilename);

std::shared_ptr<ListObject> abc_split_by_name(std::shared_ptr<PrimitiveObject> prim, bool add_when_none = false);
}

#endif //ZENO_ABCCOMMON_H
