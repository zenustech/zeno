#include "ClusterCache.h"
#include "ClusterCoverage.h"
#include "ClusterCoverageGpu.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <limits>
#include <mutex>
#include <set>

using namespace zeno::cluster;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace {
constexpr uint32_t QualityVersion = 5;
struct Analysis {
    fs::path metadata;
    nlohmann::json report;
};
uint32_t unsignedArgument(const std::string& value) {
    size_t used = 0;
    const auto number = std::stoull(value, &used);
    if (value.empty() || value.front() == '-' || used != value.size() || number > UINT32_MAX)
        throw std::runtime_error("Invalid unsigned argument: " + value);
    return uint32_t(number);
}
bool insideDirectory(const fs::path& candidate, const fs::path& directory) {
    // Compare existing ancestors through the filesystem: this also handles
    // Windows drive/path casing and junctions, unlike a string-prefix check.
    for (auto path = fs::weakly_canonical(candidate);;) {
        if (fs::exists(path) && fs::equivalent(path, directory))
            return true;
        const auto parent = path.parent_path();
        if (parent == path || parent.empty())
            return false;
        path = parent;
    }
}
// A normal deviation expressed as an object-space footprint, not a bound on
// radiance error. Like the source bake's attribute metric, it refines large
// triangles with poor attributes while allowing subpixel detail to disappear.
double normalFootprint(const Vertex* const v[3], double weight) {
    double a[3], b[3], ng[3];
    for (int j = 0; j < 3; ++j) {
        a[j] = double(v[1]->position[j]) - v[0]->position[j];
        b[j] = double(v[2]->position[j]) - v[0]->position[j];
    }
    ng[0] = a[1] * b[2] - a[2] * b[1];
    ng[1] = a[2] * b[0] - a[0] * b[2];
    ng[2] = a[0] * b[1] - a[1] * b[0];
    const double area2 = std::sqrt(ng[0] * ng[0] + ng[1] * ng[1] + ng[2] * ng[2]);
    if (area2 == 0)
        return 0; // Degenerate triangles have no shading footprint.
    double worstDot = 1, maxEdge2 = 0;
    for (int k = 0; k < 3; ++k) {
        double d = 0, normalLength2 = 0, edge2 = 0;
        for (int j = 0; j < 3; ++j) {
            const double n = v[k]->normal[j];
            d += ng[j] / area2 * n;
            normalLength2 += n * n;
            const double e = double(v[(k + 1) % 3]->position[j]) - v[k]->position[j];
            edge2 += e * e;
        }
        worstDot = std::min(worstDot, normalLength2 > 0 ? d / std::sqrt(normalLength2) : -1.);
        maxEdge2 = std::max(maxEdge2, edge2);
    }
    const double error = weight * std::sqrt(maxEdge2 * 2 * (1 - std::clamp(worstDot, -1., 1.)));
    if (!std::isfinite(error) || error > std::numeric_limits<float>::max())
        throw std::runtime_error("Non-finite normal-quality footprint");
    return error;
}
double triangleArea(const Vertex* const v[3]) {
    double a[3], b[3], n[3];
    for (int j = 0; j < 3; ++j) {
        a[j] = double(v[1]->position[j]) - v[0]->position[j];
        b[j] = double(v[2]->position[j]) - v[0]->position[j];
    }
    n[0] = a[1]*b[2]-a[2]*b[1]; n[1] = a[2]*b[0]-a[0]*b[2]; n[2] = a[0]*b[1]-a[1]*b[0];
    return .5 * std::sqrt(n[0]*n[0]+n[1]*n[1]+n[2]*n[2]);
}
Analysis analyze(const Geometry& geometry, const fs::path& input, const fs::path& output,
                 uint32_t sourceMesh, float weight, float minAreaRatio, float coverageWeight, bool gpuCoverage, bool convexCoverage) {
    auto meta = readJson(input);
    // Normal-only enrichment takes a maximum with the existing error. It is
    // idempotent and preserves earlier coverage bounds, spheres and guards.
    const bool normalOnly = weight > 0 && minAreaRatio == 0 && coverageWeight == 0;
    if (meta.contains("lodQuality") && !normalOnly)
        throw std::runtime_error("Quality analysis must start from the original geometry index");
    const auto previousQuality = meta.value("lodQuality", nlohmann::json());
    std::vector<float> errors(geometry.groups.size());
    std::vector<double> area(errors.size()), simplifiedArea(errors.size());
    std::vector<bool> hasSimplified(errors.size());
    std::vector<std::vector<coverage::Triangle>> coarse(coverageWeight>0?errors.size():0);
    for (size_t i = 0; i < errors.size(); ++i)
        errors[i] = geometry.groups[i].bounds[4];
    std::ifstream stream(geometry.pages, std::ios::binary);
    if (!stream)
        throw std::runtime_error("Cannot open geometry payload for quality analysis");
    uint64_t triangles = 0, bytes = 0;
    for (uint32_t id = 0; id < geometry.groups.size(); ++id) {
        const auto& group = geometry.groups[id];
        if (!group.depth && minAreaRatio == 0)
            continue;
        const auto page = readPage(geometry, id, stream);
        bytes += group.bytes;
        for (const auto& cluster : page.clusters) {
            if (cluster.refined < 0 && minAreaRatio == 0)
                continue;
            if (cluster.refined >= 0 && uint32_t(cluster.refined) >= id)
                throw std::runtime_error("Quality analysis requires an ordered refinement DAG");
            for (uint32_t t = 0; t < cluster.triangleCount; ++t) {
                const Vertex* vertices[3];
                for (int j = 0; j < 3; ++j)
                    vertices[j] = &page.vertices[cluster.vertexOffset +
                        page.indices[cluster.indexOffset + t * 3 + j]];
                // This coarse cluster was GENERATED from cluster.refined.
                // Charge its error to that generating group, whose metric
                // decides whether the renderer may retain this coarse cluster.
                if (minAreaRatio > 0) {
                    const double triangle = triangleArea(vertices);
                    area[id] += triangle;
                    if (cluster.refined >= 0) {
                        simplifiedArea[cluster.refined] += triangle;
                        hasSimplified[cluster.refined] = true;
                    }
                }
                if (cluster.refined >= 0) {
                    if(coverageWeight>0)coarse[cluster.refined].push_back(coverage::triangle(vertices));
                    if (weight > 0) errors[cluster.refined] = std::max(errors[cluster.refined],
                        float(normalFootprint(vertices, weight)));
                    ++triangles;
                }
            }
        }
    }
    uint32_t coverageGroups=0;double maxCoverage=0;
    std::vector<double> accumulated(errors.size()), localBounds(errors.size());
    if(coverageWeight>0) {
        std::unique_ptr<coverage::GpuBatch> batch;
        if(gpuCoverage)batch=std::make_unique<coverage::GpuBatch>(convexCoverage);
        for(uint32_t id=0;id<geometry.groups.size();++id) {
            if(coarse[id].empty())continue;
            coverage::Surface surface(std::move(coarse[id]));
            const auto page=readPage(geometry,id,stream);bytes+=geometry.groups[id].bytes;
            std::vector<coverage::Triangle> fine;
            if(batch)fine.reserve(geometry.groups[id].triangles);
            for(const auto& cluster:page.clusters)for(uint32_t t=0;t<cluster.triangleCount;++t) {
                const Vertex* v[3];for(int j=0;j<3;++j)v[j]=&page.vertices[cluster.vertexOffset+page.indices[cluster.indexOffset+t*3+j]];
                const auto triangle=coverage::triangle(v);
                if(batch)fine.push_back(triangle);
                else localBounds[id]=std::max(localBounds[id],convexCoverage?surface.convexBound(triangle):surface.bound(triangle));
            }
            if(batch) {
                batch->add(id,surface,fine);
                if(batch->triangles()>=262144)batch->flush(localBounds);
            }
            ++coverageGroups;
        }
        if(batch)batch->flush(localBounds);
    }
    if(coverageWeight>0)for(uint32_t id=0;id<geometry.groups.size();++id) {
        double inherited=0;
        for(int32_t child:geometry.groups[id].refined)if(child>=0)inherited=std::max(inherited,accumulated[child]);
        const double local=localBounds[id];
        // Triangle inequality accumulates directed coverage distance across
        // simplification levels; max-only merging would understate fine-to-LOD loss.
        accumulated[id]=inherited+local;maxCoverage=std::max(maxCoverage,accumulated[id]);
        meta["groups"][id]["coverageBound"]=accumulated[id];
        // The metric's distance denominator must also contain the finest
        // descendants, which can extend beyond this LOD's current surface.
        if(inherited>0) {
            const double radius=double(geometry.groups[id].bounds[3])+inherited;
            if(!std::isfinite(radius)||radius>std::numeric_limits<float>::max())throw std::runtime_error("Coverage sphere overflow");
            meta["groups"][id]["bounds"][3]=std::min(std::nextafter(float(radius),std::numeric_limits<float>::infinity()),std::numeric_limits<float>::max());
        }
        const double bound=coverageWeight*accumulated[id];
        if(!std::isfinite(bound)||bound>std::numeric_limits<float>::max())throw std::runtime_error("Coverage error overflow");
        const float rounded=std::min(std::nextafter(float(bound),std::numeric_limits<float>::infinity()),std::numeric_limits<float>::max());
        errors[id]=std::max(errors[id],bound>0?rounded:0.f);
    }
    uint32_t areaGuards = 0;
    double lowestAreaRatio = 1;
    for (size_t id = 0; id < errors.size(); ++id) {
        if (minAreaRatio == 0 || !hasSimplified[id] || area[id] == 0) continue;
        const double ratio = simplifiedArea[id] / area[id];
        lowestAreaRatio = std::min(lowestAreaRatio, ratio);
        if (ratio < minAreaRatio) {
            // Never select this destructive simplification. FLT_MAX is the
            // native terminal-group sentinel and preserves finite metadata.
            // This is an area-retention policy, not a surface-distance bound.
            errors[id] = std::numeric_limits<float>::max();
            ++areaGuards;
        }
    }
    // Increasing group IDs are topological order. Preserve the existing
    // monotonic error envelope so all parents/children select a complete cut.
    for (uint32_t id = 0; id < geometry.groups.size(); ++id)
        for (int32_t child : geometry.groups[id].refined)
            if (child >= 0)
                errors[id] = std::max(errors[id], errors[child]);
    size_t changed = 0;
    for (size_t id = 0; id < errors.size(); ++id) {
        changed += errors[id] > geometry.groups[id].bounds[4];
        meta["groups"][id]["bounds"][4] = errors[id];
    }
    meta["pages"] = fs::absolute(geometry.pages).string();
    meta["lodQuality"] = {{"version", QualityVersion}, {"normalWeight", weight},
                           {"coverageWeight",coverageWeight},
                           {"coverageSphereGuard",coverageWeight>0},
                           {"coverageMethod",convexCoverage?"convex":"tight"},
                           {"minAreaRatio", minAreaRatio},
                           {"inputMetadata", input.string()}};
    if (!previousQuality.is_null()) meta["lodQuality"]["previous"] = previousQuality;
    const auto path = output / ("mesh-" + std::to_string(sourceMesh) + ".json");
    writeJson(path, meta);
    return {path, {{"sourceMesh", sourceMesh}, {"groups", errors.size()},
                   {"changedGroups", changed}, {"analyzedTriangles", triangles}, {"bytesRead", bytes},
                   {"areaGuardGroups", areaGuards}, {"lowestAreaRatio", lowestAreaRatio},
                   {"coverageGroups",coverageGroups},{"maxCoverageBound",maxCoverage}}};
}
} // namespace

int main(int argc, char** argv) try {
    fs::path input, output;
    float weight = .5f;
    float minAreaRatio = 0;
    float coverageWeight=0;
    bool gpuCoverage=false;
    bool convexCoverage=false;
    uint32_t jobs = 4;
    std::set<uint32_t> selected;
    for (int i = 1; i < argc; ++i) {
        if (i + 1 >= argc)
            throw std::runtime_error("Missing quality command argument");
        const std::string arg = argv[i], value = argv[++i];
        if (arg == "--input") input = value;
        else if (arg == "--output") output = value;
        else if(arg=="--coverage-device") {
            if(value!="cpu"&&value!="cuda")throw std::runtime_error("Coverage device must be cpu or cuda");
            gpuCoverage=value=="cuda";
        }
        else if(arg=="--coverage-method") {
            if(value!="tight"&&value!="convex")throw std::runtime_error("Coverage method must be tight or convex");
            convexCoverage=value=="convex";
        }
        else if (arg == "--normal-weight" || arg == "--min-area-ratio" || arg=="--coverage-weight") {
            size_t used = 0;
            const float parsed = std::stof(value, &used);
            if (used != value.size()) throw std::runtime_error("Invalid quality weight");
            (arg == "--normal-weight" ? weight : arg=="--coverage-weight"?coverageWeight:minAreaRatio) = parsed;
        }
        else if (arg == "--jobs") jobs = unsignedArgument(value);
        else if (arg == "--mesh") selected.insert(unsignedArgument(value));
        else throw std::runtime_error("Unknown quality option: " + arg);
    }
    if (input.empty() || output.empty())
        throw std::runtime_error("Usage: zeno_cluster_quality --input asset.json --output new-index-directory "
                                 "[--normal-weight 0.5] [--min-area-ratio 0] [--coverage-weight 0] "
                                 "[--coverage-device cpu|cuda] [--coverage-method tight|convex] [--jobs 4] [--mesh source-id ...]");
    if (!std::isfinite(weight) || weight < 0 || weight > 4 || !std::isfinite(minAreaRatio) ||
        minAreaRatio < 0 || minAreaRatio > 1 || !std::isfinite(coverageWeight)||coverageWeight<0||coverageWeight>4||
        (weight == 0 && minAreaRatio == 0 && coverageWeight==0) || jobs < 1 || jobs > 32)
        throw std::runtime_error("Normal and coverage weights must be in [0,4], area ratio in [0,1], at least one positive; jobs in [1,32]");
    input = fs::absolute(input);
    output = fs::weakly_canonical(fs::absolute(output));
    if (fs::exists(output))
        throw std::runtime_error("Quality output directory must be new; existing indices are never overwritten");
    const auto start = Clock::now();
    auto manifest = readJson(input);
    const auto asset = loadAsset(input);
    if (asset.geometries.empty() || asset.placements.empty())
        throw std::runtime_error("Input has no renderable geometry");
    const bool normalOnly = weight > 0 && minAreaRatio == 0 && coverageWeight == 0;
    const auto previousQuality = manifest.value("lodQuality", nlohmann::json());
    if (manifest.contains("lodQuality") && selected.empty() && !normalOnly)
        throw std::runtime_error("Input is already a quality index; use the original asset index");
    std::vector<bool> process(asset.geometries.size());
    std::set<uint32_t> found, uniqueMeshes;
    const auto& sourceMeshes = static_cast<const nlohmann::json&>(manifest).at("meshes");
    for (size_t g = 0; g < asset.geometries.size(); ++g) {
        uint32_t sourceMesh = manifest["meshes"][g]["sourceMesh"];
        if (!uniqueMeshes.insert(sourceMesh).second)
            throw std::runtime_error("Quality analysis requires unique source mesh IDs");
        process[g] = selected.empty() || selected.count(sourceMesh);
        if (process[g]) {
            found.insert(sourceMesh);
            if (manifest.contains("lodQuality")) {
                auto path = fs::path(sourceMeshes.at(g).at("metadata").get<std::string>());
                if (path.is_relative()) path = input.parent_path() / path;
                if (readJson(path).contains("lodQuality") && !normalOnly)
                    throw std::runtime_error("Selected mesh is already a quality index; start from its original metadata");
            }
        }
    }
    if (!selected.empty() && selected != found)
        throw std::runtime_error("Selected source mesh is not present in this asset index");
    // Payloads are read-only. Do not create even metadata inside their native
    // asset directory (or overwrite a payload through a filename collision).
    auto source = fs::path(manifest.at("source").get<std::string>());
    if (source.is_relative()) {
        source = fs::absolute(input.parent_path() / source);
        // The output index lives elsewhere; retain a usable source reference.
        manifest["source"] = source.string();
    }
    source = fs::weakly_canonical(source.parent_path());
    if (insideDirectory(output, source))
        throw std::runtime_error("Quality output must be outside the source asset directory");
    fs::create_directories(output);
    std::vector<Analysis> results(asset.geometries.size());
    std::atomic<size_t> next{0}, done{0};
    std::mutex logMutex;
    std::vector<std::future<void>> workers;
    for (uint32_t w = 0; w < std::min<size_t>(jobs, asset.geometries.size()); ++w)
        workers.push_back(std::async(std::launch::async, [&] {
            for (;;) {
                const size_t g = next++;
                if (g >= asset.geometries.size()) break;
                auto path = fs::path(sourceMeshes.at(g).at("metadata").get<std::string>());
                if (path.is_relative()) path = input.parent_path() / path;
                if (process[g])
                    results[g] = analyze(asset.geometries[g], fs::absolute(path), output,
                                         sourceMeshes.at(g).at("sourceMesh"), weight, minAreaRatio,coverageWeight,gpuCoverage,convexCoverage);
                else results[g].metadata = fs::absolute(path);
                const auto count = ++done;
                if (count % 32 == 0 || count == asset.geometries.size()) {
                    std::lock_guard<std::mutex> lock(logMutex);
                    std::cout << "[quality] geometries=" << count << '/' << asset.geometries.size()
                              << " elapsed=" << std::chrono::duration<double>(Clock::now()-start).count()
                              << " s\n" << std::flush;
                }
            }
        }));
    std::exception_ptr failure;
    for (auto& worker : workers)
        try { worker.get(); } catch (...) { if (!failure) failure = std::current_exception(); }
    if (failure) std::rethrow_exception(failure);
    nlohmann::json reports = nlohmann::json::array();
    for (size_t g = 0; g < results.size(); ++g) {
        manifest["meshes"][g]["metadata"] = results[g].metadata.string();
        if (process[g]) reports.push_back(results[g].report);
    }
    manifest["lodQuality"] = {{"version", QualityVersion}, {"normalWeight", weight},
                               {"coverageWeight",coverageWeight},
                               {"coverageSphereGuard",coverageWeight>0},
                               {"coverageDevice",gpuCoverage?"cuda":"cpu"},
                               {"coverageMethod",convexCoverage?"convex":"tight"},
                               {"minAreaRatio", minAreaRatio},
                               {"inputManifest", input.string()}, {"analyzedMeshes", found.size()}};
    if (!previousQuality.is_null()) manifest["lodQuality"]["previous"] = previousQuality;
    writeJson(output/"asset.json.tmp", manifest);
    // Reopen through the production parser before publishing a usable index.
    loadAsset(output/"asset.json.tmp");
    writeJson(output/"quality-report.json", {{"settings", manifest["lodQuality"]},
        {"seconds", std::chrono::duration<double>(Clock::now()-start).count()}, {"meshes", reports}});
    fs::rename(output/"asset.json.tmp", output/"asset.json");
    std::cout << "[quality] asset=" << output/"asset.json" << '\n';
    return 0;
} catch (const std::exception& e) {
    std::cerr << "[quality] " << e.what() << '\n';
    return 1;
}
