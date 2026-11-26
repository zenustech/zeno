#include "ClusterLayerBake.h"
#include "ClusterLayerGeometry.h"
#include "ClusterSource.h"
#include <fastenvelope/FastEnvelope.h>
#include <meshoptimizer.h>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <chrono>
#include <atomic>
#include <future>
#include <thread>
#include <cstring>
#include <functional>
#include <iostream>
#include <map>
#include <mutex>
#include <set>

namespace zeno::cluster::layers {
namespace {
using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;
thread_local Scene::Context* active = nullptr;
Point transform(const glm::dmat4& m, Point p) { return Point(m * glm::dvec4(p, 1.)); }
double normBound(const glm::dmat4& m) {
    // ||M||2 <= sqrt(||transpose(M)*M||infinity). This bound stays tight for
    // rotated nonuniform scales and still bounds arbitrary affine shear.
    double squared = 0;
    for (int i = 0; i < 3; ++i) {
        double row = 0;
        for (int j = 0; j < 3; ++j) {
            double dot = 0; for (int k = 0; k < 3; ++k) dot += m[i][k] * m[j][k];
            row += std::abs(dot);
        }
        squared = std::max(squared, row);
    }
    return std::nextafter(std::sqrt(squared) * (1 + 64 * std::numeric_limits<double>::epsilon()),
                          std::numeric_limits<double>::infinity());
}
Box transformed(const Box& b, const glm::dmat4& m) {
    Box out;
    for (int i = 0; i < 8; ++i) out.add(transform(m, {i & 1 ? b.hi.x : b.lo.x, i & 2 ? b.hi.y : b.lo.y, i & 4 ? b.hi.z : b.lo.z}));
    return out;
}
std::shared_ptr<Surface> makeSurface(const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices) {
    if (indices.size() % 3) throw std::runtime_error("Layer reference has incomplete triangles");
    for (auto p : positions) if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z)) throw std::runtime_error("Nonfinite layer-reference position");
    for (auto i : indices) if (i >= positions.size()) throw std::runtime_error("Layer-reference index outside vertex buffer");
    std::vector<Point> v; v.reserve(positions.size()); for (auto p : positions) v.emplace_back(p);
    std::vector<std::array<uint32_t, 3>> f; f.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) f.push_back({indices[i], indices[i + 1], indices[i + 2]});
    return std::make_shared<Surface>(std::move(v), std::move(f));
}
uint64_t hashText(uint64_t hash, const std::string& text) {
    for (unsigned char c : text) { hash ^= c; hash *= 1099511628211ull; } return hash;
}
template<class Function> void parallelFaces(size_t count, unsigned workers, const Function& function) {
    if (workers < 2 || count < 4096) { for (size_t i = 0; i < count; ++i) function(uint32_t(i)); return; }
    std::atomic<size_t> next{0}; std::vector<std::future<void>> tasks;
    for (unsigned worker = 0; worker < workers; ++worker) tasks.push_back(std::async(std::launch::async, [&] {
        for (;;) {
            const auto begin = next.fetch_add(256); if (begin >= count) break;
            for (size_t i = begin; i < std::min(begin + 256, count); ++i) function(uint32_t(i));
        }
    }));
    std::exception_ptr error;
    for (auto& task : tasks) try { task.get(); } catch (...) { if (!error) error = std::current_exception(); }
    if (error) std::rethrow_exception(error);
}
}

struct Scene::Impl {
    struct Ref {
        size_t mesh, primitive;
        glm::dmat4 world, inverse;
        Box box;
        double scale, inverseScale;
    };
    struct Entry { std::shared_ptr<Surface> surface; uint64_t age; };
    Json json;
    std::filesystem::path root;
    std::vector<Ref> refs;
    std::unordered_map<size_t, size_t> aliases;
    std::mutex mutex;
    std::map<std::pair<size_t, size_t>, Entry> cache;
    std::unordered_map<size_t, std::string> contentKeys;
    uint64_t capacity, cacheBytes = 0, age = 0;
    unsigned workers;
    explicit Impl(const std::filesystem::path& source, uint64_t bytes, unsigned threads) : json(readJson(source)), root(source.parent_path()), capacity(bytes),
        workers(std::clamp(threads ? threads : std::thread::hardware_concurrency(), 1u, 16u)) {
        Source input{json, root, {}};
        const auto& nodes = json.at("nodes");
        std::vector<int> parents(nodes.size(), -1), state(nodes.size());
        std::vector<glm::mat4> matrices(nodes.size(), glm::mat4(1));
        for (size_t i = 0; i < nodes.size(); ++i) for (size_t c : nodes[i].value("children", std::vector<size_t>{})) {
            if (c >= nodes.size() || parents[c] != -1) throw std::runtime_error("Invalid layer-reference hierarchy");
            parents[c] = int(i);
        }
        std::function<glm::mat4(size_t)> world = [&](size_t i) -> glm::mat4 {
            if (state[i] == 2) return matrices[i];
            if (state[i] == 1) throw std::runtime_error("Cyclic layer-reference hierarchy");
            state[i] = 1; const auto& n = nodes[i]; glm::mat4 m(1);
            if (n.contains("matrix")) { auto a = n["matrix"].get<std::array<float, 16>>(); m = glm::make_mat4(a.data()); }
            else {
                const auto t = n.value("translation", std::array<float, 3>{0, 0, 0});
                const auto r = n.value("rotation", std::array<float, 4>{0, 0, 0, 1});
                const auto s = n.value("scale", std::array<float, 3>{1, 1, 1});
                m = glm::translate(glm::mat4(1), glm::make_vec3(t.data())) * glm::mat4_cast(glm::quat(r[3], r[0], r[1], r[2])) * glm::scale(glm::mat4(1), glm::make_vec3(s.data()));
            }
            matrices[i] = parents[i] < 0 ? m : world(size_t(parents[i])) * m; state[i] = 2; return matrices[i];
        };
        std::map<std::pair<size_t, size_t>, Box> boxes;
        auto add = [&](size_t mesh, const glm::mat4& matrix) {
            const glm::dmat4 m(matrix); const double det = glm::determinant(glm::dmat3(m));
            if (!std::isfinite(det)) throw std::runtime_error("Nonfinite layer-reference transform");
            if (det == 0) return; // A zero-scale placement has no visible area.
            const auto inverse = glm::inverse(m);
            const auto& primitives = json.at("meshes").at(mesh).at("primitives");
            for (size_t p = 0; p < primitives.size(); ++p) {
                const auto& primitive = primitives[p]; if (primitive.value("mode", 4) != 4) continue;
                const auto key = std::make_pair(mesh, p);
                auto found = boxes.find(key);
                if (found == boxes.end()) {
                    const auto& a = json.at("accessors").at(primitive.at("attributes").at("POSITION").get<size_t>());
                    Box b;
                    if (a.value("componentType", 0) == 5126 && a.contains("min") && a.contains("max")) {
                        const auto lo = a.at("min").get<std::array<double, 3>>(), hi = a.at("max").get<std::array<double, 3>>();
                        Point lower, upper;
                        for (int k = 0; k < 3; ++k) {
                            if (!std::isfinite(float(lo[k])) || !std::isfinite(float(hi[k])) || lo[k] > hi[k]) throw std::runtime_error("Invalid layer-reference accessor bounds");
                            lower[k] = std::nextafter(float(lo[k]), -std::numeric_limits<float>::infinity());
                            upper[k] = std::nextafter(float(hi[k]), std::numeric_limits<float>::infinity());
                        }
                        b.add(lower); b.add(upper);
                    } else { for (auto v : input.vectors(primitive.at("attributes").at("POSITION"))) b.add(Point(v)); input.decoded.clear(); }
                    found = boxes.emplace(key, b).first;
                }
                refs.push_back({mesh, p, m, inverse, transformed(found->second, m), normBound(m), normBound(inverse)});
            }
        };
        for (size_t i = 0; i < nodes.size(); ++i) {
            const auto& n = nodes[i]; if (!n.contains("mesh")) continue;
            const auto mesh = n.at("mesh").get<size_t>(); const auto m = world(i);
            const auto ext = n.value("extensions", Json::object());
            if (!ext.contains("EXT_mesh_gpu_instancing")) { add(mesh, m); continue; }
            const auto& a = ext.at("EXT_mesh_gpu_instancing").at("attributes"); size_t count = 0;
            for (auto it = a.begin(); it != a.end(); ++it) {
                const auto c = json.at("accessors").at(it.value().get<size_t>()).at("count").get<size_t>();
                if (count && count != c) throw std::runtime_error("Layer-reference instance counts differ"); count = c;
            }
            if (!count) throw std::runtime_error("Empty layer-reference instances");
            auto t = a.contains("TRANSLATION") ? input.vectors(a.at("TRANSLATION")) : std::vector<glm::vec3>(count, glm::vec3(0));
            auto s = a.contains("SCALE") ? input.vectors(a.at("SCALE")) : std::vector<glm::vec3>(count, glm::vec3(1));
            auto r = a.contains("ROTATION") ? input.rotations(a.at("ROTATION")) : std::vector<glm::quat>(count, glm::quat(1, 0, 0, 0));
            for (size_t k = 0; k < count; ++k) add(mesh, m * glm::translate(glm::mat4(1), t[k]) * glm::mat4_cast(r[k]) * glm::scale(glm::mat4(1), s[k]));
            input.decoded.clear();
        }
    }
    size_t canonical(size_t mesh) const { auto it = aliases.find(mesh); return it == aliases.end() ? mesh : it->second; }
    std::vector<size_t> poses(size_t mesh, size_t primitive) const {
        std::vector<size_t> out; for (size_t i = 0; i < refs.size(); ++i) if (canonical(refs[i].mesh) == mesh && refs[i].primitive == primitive) out.push_back(i); return out;
    }
    std::vector<size_t> neighbors(size_t pose) const {
        std::vector<size_t> out;
        const auto& own = refs[pose];
        for (size_t i = 0; i < refs.size(); ++i) {
            const auto& other = refs[i];
            if (i == pose || !own.box.overlaps(other.box)) continue;
            // Coincident placements of the same canonical primitive share
            // exactly the same LOD geometry; they are not a second layer.
            if (canonical(own.mesh) == canonical(other.mesh) && own.primitive == other.primitive && own.world == other.world) continue;
            out.push_back(i);
        }
        std::sort(out.begin(), out.end(), [&](size_t a, size_t b) {
            const auto ea = refs[a].box.hi - refs[a].box.lo, eb = refs[b].box.hi - refs[b].box.lo;
            return glm::dot(ea, ea) < glm::dot(eb, eb);
        });
        return out;
    }
    std::shared_ptr<Surface> surface(size_t mesh, size_t primitive) {
        const std::lock_guard<std::mutex> lock(mutex); const auto key = std::make_pair(mesh, primitive);
        auto found = cache.find(key);
        if (found != cache.end()) { found->second.age = ++age; return found->second.surface; }
        Source input{json, root, {}}; const auto& p = json.at("meshes").at(mesh).at("primitives").at(primitive);
        auto value = makeSurface(input.vectors(p.at("attributes").at("POSITION")), input.indices(p.at("indices")));
        const auto bytes = value->bytes();
        while (cacheBytes + bytes > capacity && !cache.empty()) {
            auto victim = cache.end();
            for (auto it = cache.begin(); it != cache.end(); ++it) if (it->second.surface.use_count() == 1 && (victim == cache.end() || it->second.age < victim->second.age)) victim = it;
            if (victim == cache.end()) break;
            cacheBytes -= victim->second.surface->bytes(); cache.erase(victim);
        }
        cacheBytes += bytes; cache.emplace(key, Entry{value, ++age}); return value;
    }
};

struct Scene::Context::Impl {
    Scene::Context* previous = nullptr;
    std::unique_ptr<fastEnvelope::FastEnvelope> envelope;
    std::vector<unsigned char> sourceLocks;
    Point origin{};
    double normalization = 1;
    uint64_t faces = 0, contacts = 0, poses = 0, neighbors = 0, calls = 0, rejected = 0, retries = 0, fallbacks = 0, checks = 0;
    double minEpsilon = 0, maxEpsilon = 0, prepareMs = 0;
};
Scene::Scene(const std::filesystem::path& source, uint64_t bytes, unsigned workers) : impl(std::make_unique<Impl>(source, bytes, workers)) {}
Scene::~Scene() = default;
void Scene::setAliases(const std::unordered_map<size_t, size_t>& aliases) { impl->aliases = aliases; }
std::string Scene::dependencyKey(size_t mesh) {
    std::set<size_t> dependencies;
    Json context = Json::array();
    for (size_t i = 0; i < impl->refs.size(); ++i) if (impl->canonical(impl->refs[i].mesh) == mesh) {
        const auto& a = impl->refs[i]; dependencies.insert(a.mesh);
        auto describe = [](const Scene::Impl::Ref& ref) {
            std::array<double, 16> matrix; std::memcpy(matrix.data(), glm::value_ptr(ref.world), sizeof(matrix));
            return Json{{"mesh", ref.mesh}, {"primitive", ref.primitive}, {"world", matrix}};
        };
        Json neighbors = Json::array();
        for (auto n : impl->neighbors(i)) { dependencies.insert(impl->refs[n].mesh); neighbors.push_back(describe(impl->refs[n])); }
        context.push_back({{"placement", describe(a)}, {"neighbors", neighbors}});
    }
    uint64_t hash = hashText(14695981039346656037ull, context.dump());
    const std::lock_guard<std::mutex> lock(impl->mutex);
    Source input{impl->json, impl->root, {}};
    for (auto id : dependencies) {
        auto found = impl->contentKeys.find(id);
        if (found == impl->contentKeys.end()) found = impl->contentKeys.emplace(id, input.contentKey(id, false)).first;
        hash = hashText(hash, std::to_string(id) + ":" + found->second + ";");
    }
    return std::to_string(hash);
}
std::unique_ptr<Scene::Context> Scene::constrain(size_t mesh, size_t primitive, const std::vector<glm::vec3>& positions, const std::vector<uint32_t>& indices) {
    const auto start = Clock::now(); auto ctx = std::make_unique<Context::Impl>();
    std::cout << "[layer bake] mesh=" << mesh << " primitive=" << primitive << " sourceFaces=" << indices.size() / 3 << std::endl;
    const auto own = makeSurface(positions, indices); const auto box = own->box(); const double diagonal = glm::length(box.hi - box.lo);
    ctx->sourceLocks.resize(positions.size()); ctx->faces = own->faces.size();
    if (!(diagonal > 0) || own->faces.empty()) return std::make_unique<Context>(std::move(ctx));
    ctx->origin = (box.lo + box.hi) * .5; ctx->normalization = 1 / diagonal;
    double scale = diagonal; for (auto p : own->vertices) scale = std::max({scale, std::abs(p.x), std::abs(p.y), std::abs(p.z)});
    const double floor = 64 * std::numeric_limits<double>::epsilon() * scale;
    std::vector<double> epsilon(own->faces.size(), diagonal);
    // Separate nonlocal folds/overlapping components of the same primitive as
    // well. Adjacent welded faces are excluded; ordinary lateral tessellation
    // is not mistaken for a second layer.
    parallelFaces(own->faces.size(), impl->workers, [&](uint32_t f) {
        epsilon[f] = std::min(epsilon[f], own->separation(own->triangle(f), 4 * diagonal, true) * .25);
    });
    std::cout << "[layer bake] mesh=" << mesh << " primitive=" << primitive << " self clearance complete" << std::endl;
    const auto poses = impl->poses(mesh, primitive); ctx->poses = poses.size();
    for (auto pose : poses) {
        const auto& a = impl->refs[pose];
        for (auto target : impl->neighbors(pose)) {
            const auto& b = impl->refs[target]; ++ctx->neighbors;
            std::shared_ptr<Surface> foreign;
            // Resolve the shared reader once before launching independent
            // queries; cache loading and field writes never race.
            for (uint32_t f = 0; f < own->faces.size(); ++f) {
                if (epsilon[f] <= floor) continue;
                auto t = own->triangle(f); for (auto& p : t) p = transform(a.world, p);
                const double cap = 4 * epsilon[f] * a.scale;
                if (bounds(t).distance2(b.box) <= cap * cap) { foreign = impl->surface(b.mesh, b.primitive); break; }
            }
            if (!foreign) continue;
            parallelFaces(own->faces.size(), impl->workers, [&](uint32_t f) {
                if (epsilon[f] <= floor) return;
                const auto t = own->triangle(f);
                Triangle worldTri;
                for (int k = 0; k < 3; ++k) worldTri[k] = transform(a.world, t[k]);
                const double worldCap = 4 * epsilon[f] * a.scale;
                if (bounds(worldTri).distance2(b.box) > worldCap * worldCap) return;
                const double lower = foreign->separationWorld(worldTri, b.world, b.inverse, b.inverseScale, worldCap);
                epsilon[f] = std::min(epsilon[f], lower / (4 * a.scale));
            });
            std::cout << "[layer bake] mesh=" << mesh << " primitive=" << primitive << " neighborPairs=" << ctx->neighbors
                      << " targetMesh=" << b.mesh << " targetPrimitive=" << b.primitive << std::endl;
        }
    }
    for (uint32_t f = 0; f < own->faces.size(); ++f) {
        if (epsilon[f] <= floor) { ++ctx->contacts; for (auto v : own->faces[f]) ctx->sourceLocks[v] |= meshopt_SimplifyVertex_Lock; }
        epsilon[f] = std::max(floor, epsilon[f]);
    }
    ctx->minEpsilon = *std::min_element(epsilon.begin(), epsilon.end()); ctx->maxEpsilon = *std::max_element(epsilon.begin(), epsilon.end());
    // The exact predicates handle containment, while unit-sized coordinates
    // keep the library's degeneracy heuristics independent of asset units.
    std::vector<fastEnvelope::Vector3> v; v.reserve(positions.size());
    for (auto p : positions) { const Point q = (Point(p) - ctx->origin) * ctx->normalization; v.emplace_back(q.x, q.y, q.z); }
    for (auto& e : epsilon) e *= ctx->normalization;
    std::vector<fastEnvelope::Vector3i> f; f.reserve(indices.size() / 3);
    for (size_t i = 0; i < indices.size(); i += 3) f.emplace_back(int(indices[i]), int(indices[i + 1]), int(indices[i + 2]));
    // Use a fresh instance for each primitive: upstream's init appends prism
    // planes when called again on an existing, nonempty envelope.
    ctx->envelope = std::make_unique<fastEnvelope::FastEnvelope>(); ctx->envelope->init(v, f, epsilon);
    ctx->prepareMs = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    std::cout << "[layer bake] mesh=" << mesh << " primitive=" << primitive << " preparedMs=" << ctx->prepareMs
              << " contactFaces=" << ctx->contacts << std::endl;
    return std::make_unique<Context>(std::move(ctx));
}
Scene::Context::Context(std::unique_ptr<Impl> value) : impl(std::move(value)) { impl->previous = active; active = this; }
Scene::Context::~Context() { active = impl->previous; }
const std::vector<unsigned char>& Scene::Context::locks() const { return impl->sourceLocks; }
Json Scene::Context::statistics() const {
    return {{"version", PolicyVersion}, {"faces", impl->faces}, {"contactFaces", impl->contacts}, {"placements", impl->poses}, {"neighborPairs", impl->neighbors},
        {"minEpsilon", impl->minEpsilon}, {"maxEpsilon", impl->maxEpsilon}, {"prepareMs", impl->prepareMs},
        {"simplificationCalls", impl->calls}, {"rejectedCandidates", impl->rejected}, {"retries", impl->retries},
        {"localFallbacks", impl->fallbacks}, {"triangleChecks", impl->checks}};
}

size_t simplify(unsigned int* destination, const unsigned int* indices, size_t indexCount,
    const float* positions, size_t vertexCount, size_t positionStride, const float* attributes, size_t attributeStride,
    const float* weights, size_t attributeCount, const unsigned char* locks, size_t targetCount, float targetError,
    unsigned int options, float* resultError) {
    if (!active || !active->impl->envelope) return meshopt_simplifyWithAttributes(destination, indices, indexCount, positions, vertexCount, positionStride,
        attributes, attributeStride, weights, attributeCount, locks, targetCount, targetError, options, resultError);
    auto& ctx = *active->impl; ++ctx.calls;
    std::vector<unsigned int> ids(indices, indices + indexCount); std::sort(ids.begin(), ids.end()); ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
    std::vector<glm::vec3> localPositions; localPositions.reserve(ids.size());
    std::vector<float> localAttributes(ids.size() * attributeCount);
    std::vector<unsigned char> localLocks(ids.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        const auto id = ids[i]; const auto* p = reinterpret_cast<const float*>(reinterpret_cast<const unsigned char*>(positions) + id * positionStride);
        localPositions.emplace_back(p[0], p[1], p[2]); localLocks[i] = locks ? locks[id] : 0;
        if (attributeCount) std::memcpy(localAttributes.data() + i * attributeCount,
            reinterpret_cast<const unsigned char*>(attributes) + id * attributeStride, attributeCount * sizeof(float));
    }
    std::vector<uint32_t> localIndices(indexCount), candidate(indexCount);
    for (size_t i = 0; i < indexCount; ++i) localIndices[i] = uint32_t(std::lower_bound(ids.begin(), ids.end(), indices[i]) - ids.begin());
    const auto input = makeSurface(localPositions, localIndices);
    auto point = [&](uint32_t i) {
        const Point p = (Point(localPositions[i]) - ctx.origin) * ctx.normalization;
        return fastEnvelope::Vector3(p.x, p.y, p.z);
    };
    for (unsigned attempt = 0; attempt < 8; ++attempt) {
        const auto count = meshopt_simplifyWithAttributes(candidate.data(), localIndices.data(), localIndices.size(),
            glm::value_ptr(localPositions[0]), localPositions.size(), sizeof(glm::vec3),
            localAttributes.empty() ? nullptr : localAttributes.data(), attributeCount * sizeof(float), weights, attributeCount,
            localLocks.data(), targetCount, targetError, options & ~meshopt_SimplifySparse, resultError);
        std::vector<unsigned char> retained(ids.size());
        for (size_t i = 0; i < count; ++i) retained[candidate[i]] = 1;
        bool unsafe = false, progress = false;
        for (size_t t = 0; t < count; t += 3) {
            const std::array<fastEnvelope::Vector3, 3> tri{point(candidate[t]), point(candidate[t + 1]), point(candidate[t + 2])}; ++ctx.checks;
            if (!ctx.envelope->is_outside(tri)) continue;
            unsafe = true;
            // Locate source vertices that the rejected chord removed, rather
            // than locking its already-retained endpoints. All queries use
            // the current input group, so the new locks exist at this level.
            for (int a = 0; a <= 4; ++a) for (int b = 0; b <= 4 - a; ++b) {
                const fastEnvelope::Vector3 p = (double(a) * tri[0] + double(b) * tri[1] + double(4 - a - b) * tri[2]) / 4.;
                if (!ctx.envelope->is_outside(p)) continue;
                const auto nearest = input->closest(ctx.origin + Point(p.x(), p.y(), p.z()) / ctx.normalization);
                uint32_t selected = UINT32_MAX;
                double best = std::numeric_limits<double>::infinity();
                for (auto v : input->faces[nearest.second]) {
                    if (localLocks[v] & meshopt_SimplifyVertex_Lock) continue;
                    const Point delta = Point(localPositions[v]) - nearest.first;
                    const double distance = glm::dot(delta, delta);
                    if (selected == UINT32_MAX || (retained[selected] && !retained[v]) ||
                        (retained[selected] == retained[v] && distance < best)) { selected = v; best = distance; }
                }
                if (selected != UINT32_MAX) { localLocks[selected] |= meshopt_SimplifyVertex_Lock; progress = true; }
            }
        }
        if (!unsafe) { for (size_t i = 0; i < count; ++i) destination[i] = ids[candidate[i]]; return count; }
        ++ctx.rejected;
        if (!progress) break;
        ++ctx.retries;
    }
    // Fail closed for this partition only. Keeping its already-certified
    // input lets other partitions and later hierarchy levels keep reducing.
    ++ctx.fallbacks; std::copy(indices, indices + indexCount, destination); if (resultError) *resultError = 0;
    return indexCount;
}
} // namespace zeno::cluster::layers
