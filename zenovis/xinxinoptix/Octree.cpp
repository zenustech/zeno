#include "Octree.h"

#include <zeno/utils/vec.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "XAS.h" 
#include "TypeCaster.h"

static constexpr double kLeafAverageQuantizationScale = 4096.0;

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
    float average = 0.0f;
    uint32_t childStart = 0;
    uint8_t childMask = 0;
};

struct BuildLevel {
    int res = 0;
    std::vector<BuildNode> nodes;
};

struct CellKey {
    int32_t x = 0;
    int32_t y = 0;
    int32_t z = 0;

    bool operator==(const CellKey& that) const {
        return x == that.x && y == that.y && z == that.z;
    }
};

struct CellKeyHash {
    size_t operator()(const CellKey& key) const noexcept {
        auto mix = [](uint32_t v) -> uint64_t {
            uint64_t x = v;
            x ^= x >> 30;
            x *= 0xbf58476d1ce4e5b9ull;
            x ^= x >> 27;
            x *= 0x94d049bb133111ebull;
            x ^= x >> 31;
            return x;
        };
        return size_t(mix(uint32_t(key.x)) ^ (mix(uint32_t(key.y)) << 1) ^ (mix(uint32_t(key.z)) << 2));
    }
};

struct SparseBuildCell {
    CellKey coord;
    BuildNode node;
};

struct SparseBuildLevel {
    int res = 0;
    std::vector<SparseBuildCell> cells;
};

struct CellAccum {
    FloatMinMax minmax;
    uint64_t quantizedSum = 0;
    uint64_t coverage = 0;

    inline bool valid() const {
        return minmax.valid();
    }

    inline void update(float value, uint64_t coveredVoxels) {
        if (value == 0.0f || coveredVoxels == 0) return;
        minmax.update(value);
        quantizedSum += uint64_t(double(value) * kLeafAverageQuantizationScale + 0.5) * coveredVoxels;
        coverage += coveredVoxels;
    }
};

using SparseAccumMap = std::unordered_map<CellKey, CellAccum, CellKeyHash>;
using SparseNodeMap = std::unordered_map<CellKey, BuildNode, CellKeyHash>;
using SparseIndexMap = std::unordered_map<CellKey, size_t, CellKeyHash>;

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

bool uniformAfterHalfQuantization(const BuildNode& node) {
    return node.minmax.valid() && toHalf(node.minmax.mini) == toHalf(node.minmax.maxi);
}

bool cellLess(const SparseBuildCell& a, const SparseBuildCell& b) {
    if (a.coord.z != b.coord.z) return a.coord.z < b.coord.z;
    if (a.coord.y != b.coord.y) return a.coord.y < b.coord.y;
    return a.coord.x < b.coord.x;
}

void mergeAccum(CellAccum& dst, const CellAccum& src)
{
    if (!src.valid()) return;
    if (dst.valid()) {
        dst.minmax.merge(src.minmax);
    } else {
        dst.minmax = src.minmax;
    }
    dst.quantizedSum += src.quantizedSum;
    dst.coverage += src.coverage;
}

OcNode makeOcNode(const BuildNode& src) {
    OcNode node {};
    node.min_d = toHalf(src.minmax.mini);
    node.max_d = toHalf(src.minmax.maxi);
    node.setChildMask(src.childMask);
    if (src.childMask == 0) {
        node.setLeafAverageBits(toHalf(src.average));
    }
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

void addValueToSparseCells(SparseAccumMap& cells, const openvdb::Coord& rootMin,
                           const openvdb::Coord& rootMax, const openvdb::Coord& leafSize,
                           int leafRes, const openvdb::CoordBBox& rawBox, float value)
{
    if (value == 0.0f) return;

    const openvdb::Coord rawMin = rawBox.min();
    const openvdb::Coord rawMax = rawBox.max();
    const int minX = std::max(rootMin.x(), rawMin.x());
    const int minY = std::max(rootMin.y(), rawMin.y());
    const int minZ = std::max(rootMin.z(), rawMin.z());
    const int maxX = std::min(rootMax.x(), rawMax.x() + 1);
    const int maxY = std::min(rootMax.y(), rawMax.y() + 1);
    const int maxZ = std::min(rootMax.z(), rawMax.z() + 1);
    if (minX >= maxX || minY >= maxY || minZ >= maxZ) return;

    const int bx = std::max(0, (minX - rootMin.x()) / leafSize.x());
    const int by = std::max(0, (minY - rootMin.y()) / leafSize.y());
    const int bz = std::max(0, (minZ - rootMin.z()) / leafSize.z());
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
                const uint64_t coverage = uint64_t(ox) * uint64_t(oy) * uint64_t(oz);
                cells[CellKey { int32_t(x), int32_t(y), int32_t(z) }].update(value, coverage);
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
        leaves[i].average = float(double(cell.quantizedSum) / (kLeafAverageQuantizationScale * double(cellVolume)));
    });
    return leaves;
}

SparseBuildLevel collectBottomLeavesSparse(const openvdb::FloatGrid& vgrid,
                                           const openvdb::Coord& rootMin,
                                           const openvdb::Coord& rootMax,
                                           const openvdb::Coord& leafSize,
                                           int leafRes)
{
    using TreeT = openvdb::FloatTree;
    using LeafT = typename TreeT::LeafNodeType;

    SparseAccumMap accum;

    auto tileIt = vgrid.tree().cbeginValueOn();
    tileIt.setMaxDepth(tileIt.getLeafDepth() - 1);
    for (auto it = tileIt; it; ++it) {
        if (!it.isTileValue()) continue;
        addValueToSparseCells(accum, rootMin, rootMax, leafSize, leafRes,
                              it.getBoundingBox(), it.getValue());
    }

    std::vector<const LeafT*> leafNodes;
    for (auto it = vgrid.tree().cbeginLeaf(); it; ++it) {
        leafNodes.push_back(&(*it));
    }

    tbb::enumerable_thread_specific<SparseAccumMap> localAccums;
    tbb::parallel_for(size_t(0), leafNodes.size(), [&](size_t leafIndex) {
        auto& local = localAccums.local();
        const LeafT& leaf = *leafNodes[leafIndex];
        for (auto iter = leaf.cbeginValueOn(); iter != leaf.cendValueOn(); ++iter) {
            const float value = iter.getValue();
            if (value == 0.0f) continue;

            const auto coord = iter.getCoord();
            if (coord.x() < rootMin.x() || coord.y() < rootMin.y() || coord.z() < rootMin.z() ||
                coord.x() >= rootMax.x() || coord.y() >= rootMax.y() || coord.z() >= rootMax.z()) {
                continue;
            }

            const CellKey key {
                int32_t((coord.x() - rootMin.x()) / leafSize.x()),
                int32_t((coord.y() - rootMin.y()) / leafSize.y()),
                int32_t((coord.z() - rootMin.z()) / leafSize.z())
            };
            local[key].update(value, 1u);
        }
    });

    for (const auto& local : localAccums) {
        for (const auto& [key, cell] : local) {
            mergeAccum(accum[key], cell);
        }
    }

    const uint64_t cellVolume =
        uint64_t(leafSize.x()) * uint64_t(leafSize.y()) * uint64_t(leafSize.z());

    SparseBuildLevel level;
    level.res = leafRes;
    level.cells.reserve(accum.size());
    for (auto& [coord, cell] : accum) {
        if (cell.coverage < cellVolume) {
            cell.minmax.update(0.0f);
        }
        if (!cell.valid() || cell.minmax.maxi <= 0.0f) {
            continue;
        }

        SparseBuildCell sparseCell {};
        sparseCell.coord = coord;
        sparseCell.node.minmax = cell.minmax;
        sparseCell.node.average = float(double(cell.quantizedSum) / (kLeafAverageQuantizationScale * double(cellVolume)));
        level.cells.push_back(sparseCell);
    }

    std::sort(level.cells.begin(), level.cells.end(), cellLess);
    return level;
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
        if (activeNode(parent) && uniformAfterHalfQuantization(parent)) {
            parent.childMask = 0;
            parent.average = parent.minmax.maxi;
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

BuildLevel toBuildLevel(const SparseBuildLevel& sparse)
{
    BuildLevel level;
    level.res = sparse.res;
    level.nodes.resize(sparse.cells.size());
    tbb::parallel_for(size_t(0), sparse.cells.size(), [&](size_t i) {
        level.nodes[i] = sparse.cells[i].node;
    });
    return level;
}

SparseBuildLevel reduceParentSparse(SparseBuildLevel& compactedChildren, const SparseBuildLevel& children)
{
    SparseNodeMap parentMap;
    parentMap.reserve(children.cells.size() / 2 + 1);

    for (const auto& childCell : children.cells) {
        if (!activeNode(childCell.node)) continue;

        const CellKey parentKey {
            int32_t(childCell.coord.x >> 1),
            int32_t(childCell.coord.y >> 1),
            int32_t(childCell.coord.z >> 1)
        };
        const uint8_t slot = uint8_t((childCell.coord.x & 1) | ((childCell.coord.y & 1) << 1) | ((childCell.coord.z & 1) << 2));
        auto& parent = parentMap[parentKey];
        parent.childMask |= uint8_t(1u << slot);
        parent.minmax.merge(childCell.node.minmax);
    }

    SparseBuildLevel parents;
    parents.res = children.res >> 1;
    parents.cells.reserve(parentMap.size());
    for (auto& [coord, node] : parentMap) {
        if (!activeNode(node)) continue;
        if (uniformAfterHalfQuantization(node)) {
            node.childMask = 0;
            node.average = node.minmax.maxi;
        }
        parents.cells.push_back(SparseBuildCell { coord, node });
    }
    std::sort(parents.cells.begin(), parents.cells.end(), cellLess);

    SparseIndexMap childIndex;
    childIndex.reserve(children.cells.size());
    for (size_t i = 0; i < children.cells.size(); ++i) {
        childIndex.emplace(children.cells[i].coord, i);
    }

    compactedChildren.res = children.res;
    compactedChildren.cells.clear();
    compactedChildren.cells.reserve(children.cells.size());
    for (auto& parentCell : parents.cells) {
        BuildNode& parent = parentCell.node;
        parent.childStart = uint32_t(compactedChildren.cells.size());
        for (uint8_t slot = 0; slot < 8; ++slot) {
            const uint8_t bit = uint8_t(1u << slot);
            if ((parent.childMask & bit) == 0) continue;

            const CellKey childKey {
                int32_t(parentCell.coord.x * 2 + ((slot >> 0) & 1)),
                int32_t(parentCell.coord.y * 2 + ((slot >> 1) & 1)),
                int32_t(parentCell.coord.z * 2 + ((slot >> 2) & 1))
            };
            const auto it = childIndex.find(childKey);
            if (it != childIndex.end()) {
                compactedChildren.cells.push_back(children.cells[it->second]);
            }
        }
    }

    return parents;
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
    SparseBuildLevel current;
    current.res = leafRes;
    current = collectBottomLeavesSparse(vgrid, rootMin, rootMax, leafSize, leafRes);

    if (current.cells.empty()) {
        octree.clear();
        return;
    }

    for (int depth = buildDepth - 1; depth >= 0; --depth) {
        SparseBuildLevel compactedChildren;
        SparseBuildLevel parents = reduceParentSparse(compactedChildren, current);
        sparseLevels[depth + 1] = toBuildLevel(compactedChildren);
        current = std::move(parents);
    }

    if (!current.cells.empty() && activeNode(current.cells[0].node)) {
        sparseLevels[0] = toBuildLevel(current);
    }

    emitSparseLevels(octree, sparseLevels);
}

void VolumeAggregate::aggregate(openvdb::FloatGrid& vgrid, int buildDepth)
{
    aggregate(vgrid, vgrid.evalActiveVoxelBoundingBox(), buildDepth);
}

void VolumeAggregate::aggregate(openvdb::FloatGrid& vgrid, const openvdb::CoordBBox& bbox, int buildDepth)
{
    octbox = {};
    octree = {};
    octreeBuildDepth = OCTREE_DEPTH;

    const auto dim = bbox.dim();
    const unsigned int requestedDepth = buildDepth > 0
        ? unsigned(buildDepth)
        : BAKED_SPARSE_VOLUME_DEFAULT_OCTREE_DEPTH;
    const unsigned int clampedDepth = bakedSparseVolumeClampOctreeBuildDepth(requestedDepth);
    buildDepth = int(clampedDepth);
    octreeBuildDepth = buildDepth;
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
