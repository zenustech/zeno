#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>
#include <glm/glm.hpp>

namespace zeno::cluster::layers {
using Point = glm::dvec3;
using Triangle = std::array<Point, 3>;

struct Box {
    Point lo{std::numeric_limits<double>::infinity()}, hi{-std::numeric_limits<double>::infinity()};
    void add(Point p) { lo = glm::min(lo, p); hi = glm::max(hi, p); }
    void add(const Box& b) { add(b.lo); add(b.hi); }
    bool overlaps(const Box& b) const {
        return lo.x <= b.hi.x && b.lo.x <= hi.x && lo.y <= b.hi.y && b.lo.y <= hi.y && lo.z <= b.hi.z && b.lo.z <= hi.z;
    }
    double distance2(const Box& b) const {
        const Point d = glm::max(glm::max(lo - b.hi, b.lo - hi), Point(0));
        return glm::dot(d, d);
    }
};
inline Box bounds(const Triangle& t) { Box b; for (auto p : t) b.add(p); return b; }
inline bool same(Point a, Point b) { return a.x == b.x && a.y == b.y && a.z == b.z; }
inline Point segmentPoint(Point p, Point a, Point b) {
    const auto e = b - a; const double d = glm::dot(e, e);
    return a + e * (d > 0 ? std::clamp(glm::dot(p - a, e) / d, 0., 1.) : 0.);
}
inline Point trianglePoint(Point p, const Triangle& t) {
    const auto a = t[0], ab = t[1] - a, ac = t[2] - a, n = glm::cross(ab, ac);
    const double area2 = glm::dot(n, n);
    if (area2 > 0) {
        const Point q = p - n * (glm::dot(p - a, n) / area2);
        const double v = glm::dot(glm::cross(q - a, ac), n) / area2;
        const double w = glm::dot(glm::cross(ab, q - a), n) / area2;
        if (v >= 0 && w >= 0 && v + w <= 1) return q;
    }
    Point best = a; double distance = glm::dot(p - a, p - a);
    for (int i = 0; i < 3; ++i) {
        const auto q = segmentPoint(p, t[i], t[(i + 1) % 3]); const double d = glm::dot(p - q, p - q);
        if (d < distance) { distance = d; best = q; }
    }
    return best;
}
struct Pair { double distance2 = std::numeric_limits<double>::infinity(); Point a{}, b{}; };
inline void consider(Pair& p, Point a, Point b) {
    const double d = glm::dot(a - b, a - b); if (d < p.distance2) p = {d, a, b};
}
inline void segmentPair(Pair& out, Point p, Point q, Point r, Point s) {
    const Point a = q - p, b = s - r, c = p - r;
    const double aa = glm::dot(a, a), bb = glm::dot(b, b), ab = glm::dot(a, b);
    const double ac = glm::dot(a, c), bc = glm::dot(b, c), det = aa * bb - ab * ab;
    double u = det > 0 ? std::clamp((ab * bc - bb * ac) / det, 0., 1.) : 0.;
    double v = bb > 0 ? (ab * u + bc) / bb : 0.;
    if (!(bb > 0)) u = aa > 0 ? std::clamp(-ac / aa, 0., 1.) : 0.;
    if (v < 0) { v = 0; u = aa > 0 ? std::clamp(-ac / aa, 0., 1.) : 0; }
    else if (v > 1) { v = 1; u = aa > 0 ? std::clamp((ab - ac) / aa, 0., 1.) : 0; }
    consider(out, p + u * a, r + v * b);
}
inline bool segmentTriangle(Point p, Point q, const Triangle& t, Point& hit) {
    const Point e1 = t[1] - t[0], e2 = t[2] - t[0], d = q - p, h = glm::cross(d, e2);
    const double det = glm::dot(e1, h);
    if (std::abs(det) <= 32 * std::numeric_limits<double>::epsilon() * std::sqrt(glm::dot(e1, e1) * glm::dot(h, h))) return false;
    const auto s = p - t[0]; const double u = glm::dot(s, h) / det;
    if (u < 0 || u > 1) return false;
    const auto c = glm::cross(s, e1); const double v = glm::dot(d, c) / det;
    if (v < 0 || u + v > 1) return false;
    const double k = glm::dot(e2, c) / det;
    if (k < 0 || k > 1) return false;
    hit = p + k * d; return true;
}
inline Pair trianglePair(const Triangle& a, const Triangle& b) {
    Pair out;
    for (int i = 0; i < 3; ++i) {
        Point hit;
        if (segmentTriangle(a[i], a[(i + 1) % 3], b, hit) || segmentTriangle(b[i], b[(i + 1) % 3], a, hit)) return {0, hit, hit};
        consider(out, a[i], trianglePoint(a[i], b));
        consider(out, trianglePoint(b[i], a), b[i]);
        for (int j = 0; j < 3; ++j) segmentPair(out, a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3]);
    }
    return out;
}

// Indexed, read-only source BVH. Coordinates remain double precision during
// clearance queries; no sampled or screen-space distance is used for the cap.
class Surface {
    struct Node { Box box; uint32_t begin{}, end{}, left{}, right{}; };
    std::vector<uint32_t> order;
    std::vector<Node> nodes;
    uint32_t build(uint32_t begin, uint32_t end) {
        Node n; n.begin = begin; n.end = end;
        for (uint32_t i = begin; i < end; ++i) n.box.add(bounds(triangle(order[i])));
        const uint32_t id = uint32_t(nodes.size()); nodes.push_back(n);
        if (end - begin > 8) {
            const Point extent = n.box.hi - n.box.lo;
            int axis = extent.y > extent.x ? 1 : 0; if (extent.z > extent[axis]) axis = 2;
            const uint32_t mid = begin + (end - begin) / 2;
            std::nth_element(order.begin() + begin, order.begin() + mid, order.begin() + end, [&](uint32_t a, uint32_t b) {
                const auto ta = triangle(a), tb = triangle(b);
                return ta[0][axis] + ta[1][axis] + ta[2][axis] < tb[0][axis] + tb[1][axis] + tb[2][axis];
            });
            const auto left = build(begin, mid), right = build(mid, end); nodes[id].left = left; nodes[id].right = right;
        }
        return id;
    }
    void nearest(Point p, uint32_t id, Pair& best, uint32_t& face) const {
        Box pb; pb.add(p); const auto& n = nodes[id]; if (pb.distance2(n.box) > best.distance2) return;
        if (n.left) { nearest(p, n.left, best, face); nearest(p, n.right, best, face); }
        else for (uint32_t i = n.begin; i < n.end; ++i) {
            const auto index = order[i]; const auto q = trianglePoint(p, triangle(index)); const auto d = glm::dot(q - p, q - p);
            if (d < best.distance2) { best = {d, p, q}; face = index; }
        }
    }
    void distance(const Triangle& t, const Box& box, uint32_t id, double& best, bool selfLayers) const {
        const auto& n = nodes[id]; if (box.distance2(n.box) > best) return;
        if (selfLayers) {
            const auto normal = glm::cross(t[1] - t[0], t[2] - t[0]);
            const double center = glm::dot((n.box.lo + n.box.hi) * .5 - t[0], normal);
            const double radius = glm::dot((n.box.hi - n.box.lo) * .5, glm::abs(normal));
            const double extent = std::abs(center) + radius;
            if (extent * extent < .64 * box.distance2(n.box) * glm::dot(normal, normal)) return;
        }
        if (n.left) { distance(t, box, n.left, best, selfLayers); distance(t, box, n.right, best, selfLayers); }
        else for (uint32_t i = n.begin; i < n.end; ++i) {
            const auto other = triangle(order[i]);
            if (selfLayers) {
                bool shared = false; for (auto a : t) for (auto b : other) shared |= same(a, b);
                if (shared) continue;
            }
            if (selfLayers) {
                const auto na = glm::cross(t[1] - t[0], t[2] - t[0]), nb = glm::cross(other[1] - other[0], other[2] - other[0]);
                const double aa = glm::dot(na, na), bb = glm::dot(nb, nb), ab = glm::dot(na, nb);
                if (!(aa > 0 && bb > 0) || ab * ab < .64 * aa * bb) continue;
            }
            const auto pair = trianglePair(t, other);
            if (selfLayers) {
                const auto na = glm::cross(t[1] - t[0], t[2] - t[0]), nb = glm::cross(other[1] - other[0], other[2] - other[0]);
                const double aa = glm::dot(na, na), bb = glm::dot(nb, nb);
                const auto delta = pair.a - pair.b;
                if (pair.distance2 > 0 && (std::pow(glm::dot(delta, na), 2) < .64 * pair.distance2 * aa ||
                                          std::pow(glm::dot(delta, nb), 2) < .64 * pair.distance2 * bb)) continue;
            }
            best = std::min(best, pair.distance2);
        }
    }
    void worldDistance(const Triangle& t, const Box& localBox, const glm::dmat4& world,
                       double inverseNorm2, uint32_t id, double& best) const {
        const auto& n = nodes[id]; if (localBox.distance2(n.box) > best * inverseNorm2) return;
        if (n.left) {
            worldDistance(t, localBox, world, inverseNorm2, n.left, best);
            worldDistance(t, localBox, world, inverseNorm2, n.right, best);
        } else for (uint32_t i = n.begin; i < n.end; ++i) {
            auto other = triangle(order[i]); for (auto& p : other) p = Point(world * glm::dvec4(p, 1));
            best = std::min(best, trianglePair(t, other).distance2);
        }
    }
public:
    std::vector<Point> vertices;
    std::vector<std::array<uint32_t, 3>> faces;
    Surface(std::vector<Point> v, std::vector<std::array<uint32_t, 3>> f) : vertices(std::move(v)), faces(std::move(f)) {
        order.resize(faces.size()); std::iota(order.begin(), order.end(), 0u);
        if (!order.empty()) { nodes.reserve(faces.size() / 2 + 1); build(0, uint32_t(order.size())); }
    }
    Triangle triangle(uint32_t id) const { const auto& t = faces[id]; return {vertices[t[0]], vertices[t[1]], vertices[t[2]]}; }
    Box box() const { return nodes.empty() ? Box{} : nodes[0].box; }
    uint64_t bytes() const { return vertices.capacity() * sizeof(Point) + faces.capacity() * sizeof(faces[0]) +
        nodes.capacity() * sizeof(Node) + order.capacity() * sizeof(uint32_t); }
    std::pair<Point, uint32_t> closest(Point p) const {
        Pair best; uint32_t id = 0; if (!nodes.empty()) nearest(p, 0, best, id); return {best.b, id};
    }
    double separation(const Triangle& t, double cap, bool selfLayers = false) const {
        double best = cap * cap; if (!nodes.empty()) distance(t, bounds(t), 0, best, selfLayers);
        double scale = cap;
        for (auto p : t) scale = std::max({scale, std::abs(p.x), std::abs(p.y), std::abs(p.z)});
        // Round down the computed clearance rather than granting extra room.
        return std::max(0., std::sqrt(best) - 512 * std::numeric_limits<double>::epsilon() * scale);
    }
    double separationWorld(const Triangle& t, const glm::dmat4& world, const glm::dmat4& inverse,
                           double inverseNorm, double cap) const {
        Triangle local; for (int i = 0; i < 3; ++i) local[i] = Point(inverse * glm::dvec4(t[i], 1));
        double best = cap * cap;
        if (!nodes.empty()) worldDistance(t, bounds(local), world, inverseNorm * inverseNorm, 0, best);
        double scale = cap;
        for (auto p : t) scale = std::max({scale, std::abs(p.x), std::abs(p.y), std::abs(p.z)});
        return std::max(0., std::sqrt(best) - 1024 * std::numeric_limits<double>::epsilon() * scale);
    }
};
} // namespace zeno::cluster::layers
