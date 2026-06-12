#include "Octree.h"

#include <zeno/utils/vec.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <utility>
#include <vector>

#include <tbb/parallel_for.h>

#include "XAS.h" 
#include "TypeCaster.h"

struct FloatMinMax {
    float mini = +std::numeric_limits<float>::max();
    float maxi = -std::numeric_limits<float>::max();

    inline void update(float value) {
        mini = std::min(mini, value);
        maxi = std::max(maxi, value);
    }

    inline void merge(const FloatMinMax& that) {
        mini = std::min(mini, that.mini);
        maxi = std::max(maxi, that.maxi);
    }

    inline bool valid() const {
        return mini <= maxi;
    }
};

struct BuildNode {
    FloatMinMax minmax;
    uint32_t childStart = 0;
    uint8_t childMask = 0;
};

struct BuildLevel {
    int res = 0;
    std::vector<BuildNode> nodes;
};

struct CellAccum {
    FloatMinMax minmax;
    uint64_t coverage = 0;

    inline bool valid() const {
        return minmax.valid();
    }

    inline void update(float value, uint64_t coveredVoxels) {
        if (value == 0.0f || coveredVoxels == 0) return;
        minmax.update(value);
        coverage += coveredVoxels;
    }
};

int roundUpToMultiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

size_t denseIndex(int x, int y, int z, int res) {
    return (size_t(z) * size_t(res) + size_t(y)) * size_t(res) + size_t(x);
}

void decodeDenseIndex(size_t idx, int res, int& x, int& y, int& z) {
    x = int(idx % size_t(res));
    idx /= size_t(res);
    y = int(idx % size_t(res));
    z = int(idx / size_t(res));
}

int floorToMultiple(int value, int multiple) {
    const int64_t v = int64_t(value);
    const int64_t m = int64_t(multiple);
    const int64_t div = v >= 0 ? v / m : -((-v + m - 1) / m);
    return int(div * m);
}

bool activeNode(const BuildNode& node) {
    return node.minmax.valid() && node.minmax.maxi > 0.0f;
}

OcNode makeOcNode(const BuildNode& src) {
    OcNode node {};
    node.min_d = toHalf(src.minmax.mini);
    node.max_d = toHalf(src.minmax.maxi);
    node.setChildMask(src.childMask);
    return node;
}

void addValueToCells(std::vector<CellAccum>& cells, const openvdb::Coord& rootMin,
                     const openvdb::Coord& rootMax, const openvdb::Coord& leafSize,
                     int leafRes, const openvdb::CoordBBox& rawBox, float value)
{
    if (value == 0.0f) return;

    const auto& rawMin = rawBox.min();
    const auto rawMax = rawBox.max().offsetBy(1);

    const int minX = std::max(rawMin.x(), rootMin.x());
    const int minY = std::max(rawMin.y(), rootMin.y());
    const int minZ = std::max(rawMin.z(), rootMin.z());
    const int maxX = std::min(rawMax.x(), rootMax.x());
    const int maxY = std::min(rawMax.y(), rootMax.y());
    const int maxZ = std::min(rawMax.z(), rootMax.z());
    if (minX >= maxX || minY >= maxY || minZ >= maxZ) return;

    const int bx = (minX - rootMin.x()) / leafSize.x();
    const int by = (minY - rootMin.y()) / leafSize.y();
    const int bz = (minZ - rootMin.z()) / leafSize.z();
    const int ex = std::min(leafRes, (maxX - rootMin.x() + leafSize.x() - 1) / leafSize.x());
    const int ey = std::min(leafRes, (maxY - rootMin.y() + leafSize.y() - 1) / leafSize.y());
    const int ez = std::min(leafRes, (maxZ - rootMin.z() + leafSize.z() - 1) / leafSize.z());

    for (int z = bz; z < ez; ++z) {
        const int cellMinZ = rootMin.z() + z * leafSize.z();
        const int cellMaxZ = cellMinZ + leafSize.z();
        const int oz = std::min(maxZ, cellMaxZ) - std::max(minZ, cellMinZ);
        for (int y = by; y < ey; ++y) {
            const int cellMinY = rootMin.y() + y * leafSize.y();
            const int cellMaxY = cellMinY + leafSize.y();
            const int oy = std::min(maxY, cellMaxY) - std::max(minY, cellMinY);
            for (int x = bx; x < ex; ++x) {
                const int cellMinX = rootMin.x() + x * leafSize.x();
                const int cellMaxX = cellMinX + leafSize.x();
                const int ox = std::min(maxX, cellMaxX) - std::max(minX, cellMinX);
                const size_t idx = denseIndex(x, y, z, leafRes);
                const uint64_t coverage = uint64_t(ox) * uint64_t(oy) * uint64_t(oz);
                cells[idx].update(value, coverage);
            }
        }
    }
}

std::vector<BuildNode> collectBottomLeaves(const openvdb::FloatGrid& vgrid,
                                           const openvdb::Coord& rootMin,
                                           const openvdb::Coord& rootMax,
                                           const openvdb::Coord& leafSize,
                                           int leafRes)
{
    using TreeT = openvdb::FloatTree;
    using LeafT = typename TreeT::LeafNodeType;

    const size_t cellCount = size_t(leafRes) * size_t(leafRes) * size_t(leafRes);
    std::vector<CellAccum> accum(cellCount);

    auto tileIt = vgrid.tree().cbeginValueOn();
    tileIt.setMaxDepth(tileIt.getLeafDepth() - 1);
    for (auto it = tileIt; it; ++it) {
        if (!it.isTileValue()) continue;
        addValueToCells(accum, rootMin, rootMax, leafSize, leafRes,
                        it.getBoundingBox(), it.getValue());
    }

    const uint64_t cellVolume =
        uint64_t(leafSize.x()) * uint64_t(leafSize.y()) * uint64_t(leafSize.z());

    std::vector<BuildNode> leaves(cellCount);
    const auto& tree = vgrid.tree();
    constexpr int vdbLeafDim = int(LeafT::DIM);

    tbb::parallel_for(size_t(0), cellCount, [&](size_t i) {
        int ox, oy, oz;
        decodeDenseIndex(i, leafRes, ox, oy, oz);

        const int minX = rootMin.x() + ox * leafSize.x();
        const int minY = rootMin.y() + oy * leafSize.y();
        const int minZ = rootMin.z() + oz * leafSize.z();
        const int maxX = std::min(rootMax.x(), minX + leafSize.x());
        const int maxY = std::min(rootMax.y(), minY + leafSize.y());
        const int maxZ = std::min(rootMax.z(), minZ + leafSize.z());

        CellAccum cell = accum[i];
        for (int lz = floorToMultiple(minZ, vdbLeafDim); lz < maxZ; lz += vdbLeafDim) {
            for (int ly = floorToMultiple(minY, vdbLeafDim); ly < maxY; ly += vdbLeafDim) {
                for (int lx = floorToMultiple(minX, vdbLeafDim); lx < maxX; lx += vdbLeafDim) {
                    const LeafT* leaf = tree.probeConstLeaf(openvdb::Coord(lx, ly, lz));
                    if (!leaf) continue;

                    const auto& valueMask = leaf->valueMask();
                    for (auto it = valueMask.beginOn(); it; ++it) {
                        const auto offset = it.pos();
                        const auto coord = leaf->offsetToGlobalCoord(offset);
                        if (coord.x() < minX || coord.x() >= maxX ||
                            coord.y() < minY || coord.y() >= maxY ||
                            coord.z() < minZ || coord.z() >= maxZ) {
                            continue;
                        }

                        cell.update(leaf->getValue(offset), 1);
                    }
                }
            }
        }

        if (!cell.valid()) return;
        leaves[i].minmax = cell.minmax;
        if (cell.coverage < cellVolume) leaves[i].minmax.update(0.0f);
    });
    return leaves;
}

BuildLevel reduceParentDense(const BuildLevel& children)
{
    BuildLevel parents;
    parents.res = children.res >> 1;
    const size_t parentCount =
        size_t(parents.res) * size_t(parents.res) * size_t(parents.res);
    parents.nodes.resize(parentCount);

    tbb::parallel_for(size_t(0), parentCount, [&](size_t parentIdx) {
        int x, y, z;
        decodeDenseIndex(parentIdx, parents.res, x, y, z);

        BuildNode parent {};
        for (uint8_t slot = 0; slot < 8; ++slot) {
            const int cx = x * 2 + ((slot >> 0) & 1);
            const int cy = y * 2 + ((slot >> 1) & 1);
            const int cz = z * 2 + ((slot >> 2) & 1);
            const auto& child = children.nodes[denseIndex(cx, cy, cz, children.res)];
            if (!activeNode(child)) continue;
            parent.childMask |= uint8_t(1u << slot);
            parent.minmax.merge(child.minmax);
        }
        parents.nodes[parentIdx] = parent;
    });

    return parents;
}

BuildLevel compactChildLevel(BuildLevel& parents, const BuildLevel& children)
{
    std::vector<uint8_t> childCounts(parents.nodes.size(), 0);
    std::vector<uint32_t> childOffsets(parents.nodes.size(), 0);

    tbb::parallel_for(size_t(0), parents.nodes.size(), [&](size_t parentIdx) {
        const auto& parent = parents.nodes[parentIdx];
        if (!activeNode(parent) || parent.childMask == 0) return;

        uint8_t count = 0;
        for (uint8_t slot = 0; slot < 8; ++slot) {
            if (parent.childMask & uint8_t(1u << slot)) ++count;
        }
        childCounts[parentIdx] = count;
    });

    uint32_t total = 0;
    for (size_t i = 0; i < childCounts.size(); ++i) {
        childOffsets[i] = total;
        total += childCounts[i];
    }

    BuildLevel sparse;
    sparse.res = children.res;
    sparse.nodes.resize(total);

    tbb::parallel_for(size_t(0), parents.nodes.size(), [&](size_t parentIdx) {
        auto& parent = parents.nodes[parentIdx];
        if (childCounts[parentIdx] == 0) return;

        int x, y, z;
        decodeDenseIndex(parentIdx, parents.res, x, y, z);
        parent.childStart = childOffsets[parentIdx];

        uint32_t childRank = 0;
        for (uint8_t slot = 0; slot < 8; ++slot) {
            const uint8_t bit = uint8_t(1u << slot);
            if ((parent.childMask & bit) == 0) continue;

            const int cx = x * 2 + ((slot >> 0) & 1);
            const int cy = y * 2 + ((slot >> 1) & 1);
            const int cz = z * 2 + ((slot >> 2) & 1);
            const auto& child = children.nodes[denseIndex(cx, cy, cz, children.res)];
            sparse.nodes[size_t(parent.childStart) + childRank] = child;
            ++childRank;
        }
    });

    return sparse;
}

void emitSparseLevels(std::vector<OcNode>& octree,
                      const std::vector<BuildLevel>& sparseLevels)
{
    if (sparseLevels.empty() || sparseLevels[0].nodes.empty()) {
        octree.clear();
        return;
    }

    std::vector<uint32_t> levelStart(sparseLevels.size(), 0);
    for (size_t depth = 1; depth < sparseLevels.size(); ++depth) {
        levelStart[depth] = levelStart[depth - 1] + uint32_t(sparseLevels[depth - 1].nodes.size());
    }

    const size_t totalSize = size_t(levelStart.back()) + sparseLevels.back().nodes.size();
    octree.clear();
    octree.resize(totalSize);

    for (size_t depth = 0; depth < sparseLevels.size(); ++depth) {
        const auto& level = sparseLevels[depth];
        tbb::parallel_for(size_t(0), level.nodes.size(), [&](size_t nodeIdx) {
            const auto& src = level.nodes[nodeIdx];
            OcNode node = makeOcNode(src);
            if (src.childMask != 0 && depth + 1 < sparseLevels.size()) {
                node.setChildOffset(levelStart[depth + 1] + src.childStart);
            }

            octree[size_t(levelStart[depth]) + nodeIdx] = node;
        });
    }
}

void buildBottomUpSparse(std::vector<OcNode>& octree,
                         const openvdb::Coord& rootMin,
                         const openvdb::Coord& rootMax,
                         const openvdb::Coord& leafSize,
                         int buildDepth,
                         const openvdb::FloatGrid& vgrid)
{
    std::vector<BuildLevel> sparseLevels(buildDepth + 1);
    const int leafRes = 1 << buildDepth;
    BuildLevel current;
    current.res = leafRes;
    current.nodes = collectBottomLeaves(vgrid, rootMin, rootMax, leafSize, leafRes);

    bool hasVolume = false;
    for (const auto& leaf : current.nodes) {
        if (activeNode(leaf)) {
            hasVolume = true;
            break;
        }
    }
    if (!hasVolume) {
        octree.clear();
        return;
    }

    for (int depth = buildDepth - 1; depth >= 0; --depth) {
        BuildLevel parents = reduceParentDense(current);
        sparseLevels[depth + 1] = compactChildLevel(parents, current);
        current = std::move(parents);
    }

    if (!current.nodes.empty() && activeNode(current.nodes[0])) {
        sparseLevels[0].res = 1;
        sparseLevels[0].nodes.push_back(current.nodes[0]);
    }

    emitSparseLevels(octree, sparseLevels);
}

void VolumeAggregate::aggregate(openvdb::FloatGrid& vgrid, float)
{
    octbox = {};
    octree = {};

    const auto bbox = vgrid.evalActiveVoxelBoundingBox();
    const auto dim = bbox.dim();
    const int buildDepth = OCTREE_DEPTH;
    const int leafRes = 1 << buildDepth;
    const openvdb::Coord paddedDim(
        roundUpToMultiple(dim.x(), leafRes),
        roundUpToMultiple(dim.y(), leafRes),
        roundUpToMultiple(dim.z(), leafRes));
    const auto minCoord = bbox.min();
    const openvdb::Coord maxCoord(
        minCoord.x() + paddedDim.x(),
        minCoord.y() + paddedDim.y(),
        minCoord.z() + paddedDim.z());
    octbox = Box3F(minCoord.asVec3s(), maxCoord.asVec3s());

    CppTimer timer;
    timer.tick();
    const auto leafSize = openvdb::Coord( paddedDim.x()/leafRes, paddedDim.y()/leafRes, paddedDim.z()/leafRes );
    buildBottomUpSparse(octree, minCoord, maxCoord, leafSize, buildDepth, vgrid);
    timer.tock("subdivide cost");

    const auto diff = paddedDim - dim;
    std::printf("octree size = %zu padding x=%d y=%d z=%d \n",
                octree.size(), diff.x(), diff.y(), diff.z());
}
