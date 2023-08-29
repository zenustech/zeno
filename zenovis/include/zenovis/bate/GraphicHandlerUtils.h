#pragma once

#include <memory>

#include <zeno/utils/vec.h>
#include <zenovis/opengl/buffer.h>

#include <glm/glm.hpp>

#define PI 3.1415926


namespace zenovis {
namespace {

using zeno::vec3f;
using opengl::Buffer;

void drawAxis(vec3f pos, vec3f axis, vec3f color, float size, std::unique_ptr<Buffer> &vbo) {
    std::vector<vec3f> mem;

    auto end = pos + axis * size;

    mem.push_back(pos);
    mem.push_back(color);
    mem.push_back(end);
    mem.push_back(color);

    auto vertex_count = mem.size() / 2;
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    CHECK_GL(glDrawArrays(GL_LINES, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
}

void drawCone(vec3f pos, vec3f a, vec3f b, vec3f color, float size, std::unique_ptr<Buffer> &vbo) {
    std::vector<vec3f> mem;

    auto normal = normalize(cross(a, b));

    auto top = pos + size * normal;
    auto ctr = pos - size * normal;

    float r = size / 2.0f;
    auto p = ctr + r * cos(0) * a + r * sin(0) * b;
    mem.push_back(p);
    mem.push_back(color);
    for (double t = 0.01; t < 1.0; t += 0.01) {
        double theta = 2.0 * PI * t;
        auto next_p = ctr + r * cos(theta) * a + r * sin(theta) * b;
        mem.push_back(next_p);
        mem.push_back(color);
        mem.push_back(ctr);
        mem.push_back(color);
        mem.push_back(p);
        mem.push_back(color);
        mem.push_back(top);
        mem.push_back(color);
        mem.push_back(next_p);
        mem.push_back(color);
        p = next_p;
    }

    auto vertex_count = mem.size() / 2;
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
}

void drawSquare(vec3f pos, vec3f a, vec3f b, vec3f color, float size, std::unique_ptr<Buffer> &vbo) {
    std::vector<vec3f> mem;

    auto dir1 = normalize(a + b);
    auto normal = normalize(cross(a, b));
    auto dir2 = normalize(cross(dir1, normal));

    auto ctr = pos + size * 10.0f * dir1;
    auto p1 = ctr + size * dir1;
    auto p3 = ctr - size * dir1;
    auto p2 = ctr + size * dir2;
    auto p4 = ctr - size * dir2;

    mem.push_back(p1);
    mem.push_back(color);
    mem.push_back(p2);
    mem.push_back(color);
    mem.push_back(p4);
    mem.push_back(color);
    mem.push_back(p3);
    mem.push_back(color);

    auto vertex_count = mem.size() / 2;
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
}

void drawCircle(vec3f pos, vec3f a, vec3f b, vec3f color, float size, std::unique_ptr<Buffer> &vbo) {
    constexpr int point_num = 100;
    constexpr double step = 1.0 / point_num;
    std::vector<vec3f> mem;
    mem.reserve(point_num * 2);

    for (double t = 0; t < 1.0; t += step) {
        double theta = 2.0 * PI * t;
        auto p = pos + size * cos(theta) * a + size * sin(theta) * b;
        mem.push_back(p);
        mem.push_back(color);
    }

    auto vertex_count = mem.size() / 2;
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    CHECK_GL(glDrawArrays(GL_LINE_LOOP, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
}

void drawCircle(vec3f pos, vec3f a, vec3f b, vec3f color, float size, float width, std::unique_ptr<Buffer> &vbo) {
    constexpr int point_num = 100;
    constexpr double step = 1.0 / point_num;
    std::vector<vec3f> mem;
    mem.reserve(point_num * 4);

    float inner_size = size - width / 2.0;
    float outer_size = size + width / 2.0;
    for (double t = 0; t < 1.0 + step; t += step) {
        double theta = 2.0 * PI * t;
        auto p1 = pos + inner_size * cos(theta) * a + inner_size * sin(theta) * b;
        auto p2 = pos + outer_size * cos(theta) * a + outer_size * sin(theta) * b;
        mem.push_back(p1);
        mem.push_back(color);
        mem.push_back(p2);
        mem.push_back(color);
    }

    auto vertex_count = mem.size() / 2;
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
}

void drawCube(vec3f pos, vec3f a, vec3f b, vec3f color, float size, std::unique_ptr<Buffer> &vbo) {
    std::vector<vec3f> mem;

    auto n = normalize(cross(a, b));
    auto p1 = pos + size * a + size * b + size * n; // 1 1 1
    auto p2 = pos - size * a + size * b + size * n; // 0 1 1
    auto p3 = pos + size * a + size * b - size * n; // 1 1 0
    auto p4 = pos - size * a + size * b - size * n; // 0 1 0
    auto p5 = pos + size * a - size * b + size * n; // 1 0 1
    auto p6 = pos - size * a - size * b + size * n; // 0 0 1
    auto p7 = pos + size * a - size * b - size * n; // 1 0 0
    auto p8 = pos - size * a - size * b - size * n; // 0 0 0

    // top
    mem.push_back(p1);
    mem.push_back(color);
    mem.push_back(p2);
    mem.push_back(color);
    mem.push_back(p5);
    mem.push_back(color);
    mem.push_back(p6);
    mem.push_back(color);

    // back
    mem.push_back(p2);
    mem.push_back(color);
    mem.push_back(p8);
    mem.push_back(color);
    mem.push_back(p4);
    mem.push_back(color);

    // right
    mem.push_back(p2);
    mem.push_back(color);
    mem.push_back(p3);
    mem.push_back(color);
    mem.push_back(p1);
    mem.push_back(color);

    // front
    mem.push_back(p3);
    mem.push_back(color);
    mem.push_back(p5);
    mem.push_back(color);
    mem.push_back(p7);
    mem.push_back(color);

    // left
    mem.push_back(p5);
    mem.push_back(color);
    mem.push_back(p8);
    mem.push_back(color);
    mem.push_back(p6);
    mem.push_back(color);

    // bottom
    mem.push_back(p8);
    mem.push_back(color);
    mem.push_back(p4);
    mem.push_back(color);
    mem.push_back(p7);
    mem.push_back(color);
    mem.push_back(p3);
    mem.push_back(color);

    auto vertex_count = mem.size() / 2;
    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    CHECK_GL(glDrawArrays(GL_TRIANGLE_STRIP, 0, vertex_count));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);
    vbo->unbind();
}

// http://www.songho.ca/opengl/gl_sphere.html
void drawSphere(vec3f center, vec3f color, float radius, std::unique_ptr<Buffer> &vbo, std::unique_ptr<Buffer> &ibo) {
    std::vector<vec3f> mem;
    std::vector<int> idx;

    float x, y, z, xy;

    float sectorCount = 100;
    float stackCount = 100;

    float sectorStep = 2.0f * PI / sectorCount;
    float stackStep = PI / stackCount;
    float sectorAngle, stackAngle;

    for (int i = 0; i <= stackCount; ++i) {
        stackAngle = PI / 2 - i * stackStep;        // starting from pi/2 to -pi/2
        xy = radius * cosf(stackAngle);             // r * cos(u)
        z = radius * sinf(stackAngle);              // r * sin(u)

        // add (sectorCount+1) vertices per stack
        // the first and last vertices have same position and normal, but different tex coords
        for (int j = 0; j <= sectorCount; ++j) {
            sectorAngle = j * sectorStep;           // starting from 0 to 2pi

            // vertex position (x, y, z)
            x = xy * cosf(sectorAngle);             // r * cos(u) * cos(v)
            y = xy * sinf(sectorAngle);             // r * cos(u) * sin(v)
            mem.emplace_back(x + center[0], y + center[1], z + center[2]);
            mem.push_back(color);
        }
    }

    // generate CCW index list of sphere triangles
    // k1--k1+1
    // |  / |
    // | /  |
    // k2--k2+1
    int k1, k2;
    for (int i = 0; i < stackCount; ++i) {
        k1 = i * (sectorCount + 1);     // beginning of current stack
        k2 = k1 + sectorCount + 1;      // beginning of next stack

        for(int j = 0; j < sectorCount; ++j, ++k1, ++k2) {
            // 2 triangles per sector excluding first and last stacks
            // k1 => k2 => k1+1
            if (i != 0) {
                idx.emplace_back(k1);
                idx.emplace_back(k2);
                idx.emplace_back(k1 + 1);
            }

            // k1+1 => k2 => k2+1
            if (i != (stackCount-1)) {
                idx.emplace_back(k1 + 1);
                idx.emplace_back(k2);
                idx.emplace_back(k2 + 1);
            }
        }
    }

    vbo->bind_data(mem.data(), mem.size() * sizeof(mem[0]));
    ibo->bind_data(idx.data(), idx.size() * sizeof(idx[0]));

    vbo->attribute(0, sizeof(float) * 0, sizeof(float) * 6, GL_FLOAT, 3);
    vbo->attribute(1, sizeof(float) * 3, sizeof(float) * 6, GL_FLOAT, 3);

    vbo->unbind();
    CHECK_GL(glEnable(GL_CULL_FACE));
    CHECK_GL(glCullFace(GL_BACK));
    CHECK_GL(glDrawElements(GL_TRIANGLES, idx.size(), GL_UNSIGNED_INT, (void*)nullptr));
    CHECK_GL(glDisable(GL_CULL_FACE));

    vbo->disable_attribute(0);
    vbo->disable_attribute(1);

    ibo->unbind();
}

bool rightOn(glm::vec3 v1, glm::vec3 v2, glm::vec3 n) {
    return glm::dot(glm::cross(v2, v1), n) > 0;
}

bool rayIntersectSquare(glm::vec3 ray_origin, glm::vec3 ray_direction, glm::vec3 square_min, glm::vec3 square_max,
                        glm::vec3 normal, glm::mat4 ModelMatrix) {
    auto v1 = glm::normalize(square_max - square_min);
    auto v2 = glm::normalize(glm::cross(v1, normal));
    auto len = glm::length(v1) / 2.0f;
    auto ctr = (square_min + square_max) / 2.0f;
    auto tp1 = ModelMatrix * glm::vec4(ctr + v2 * len, 1);
    auto tp3 = ModelMatrix * glm::vec4(ctr - v2 * len, 1);
    auto tp2 = ModelMatrix * glm::vec4(square_max, 1);
    auto tp4 = ModelMatrix * glm::vec4(square_min, 1);
    glm::vec3 p1 = tp1 / tp1[3];
    glm::vec3 p2 = tp2 / tp2[3];
    glm::vec3 p3 = tp3 / tp3[3];
    glm::vec3 p4 = tp4 / tp4[3];

    auto e1 = p1 - p2;
    auto e2 = p2 - p3;
    auto e3 = p3 - p4;
    auto e4 = p4 - p1;
    normal = ModelMatrix * glm::vec4(normal, 0);

    // calc intersection
    auto t = glm::dot((p1 - ray_origin), normal) / glm::dot(ray_direction, normal);
    if (t <= 0) return false;
    auto intersect_p = ray_origin + t * ray_direction;

    // test if intersection in square
    bool right_on_e1 = rightOn(e1, intersect_p - p1, normal);
    bool right_on_e2 = rightOn(e2, intersect_p - p2, normal);
    bool right_on_e3 = rightOn(e3, intersect_p - p3, normal);
    bool right_on_e4 = rightOn(e4, intersect_p - p4, normal);
    if ((right_on_e1 && right_on_e2 && right_on_e3 && right_on_e4) ||
        (!right_on_e1 && !right_on_e2 && !right_on_e3 && !right_on_e4))
        return true;
    return false;
}

/**
 * test if ray intersect an OBB
 * https://github.com/opengl-tutorials/ogl/blob/master/misc05_picking/misc05_picking_custom.cpp#L83
 */
bool rayIntersectOBB(glm::vec3 ray_origin, glm::vec3 ray_direction, glm::vec3 aabb_min, glm::vec3 aabb_max,
                     glm::mat4 ModelMatrix, float &intersection_distance) {
    // Intersection method from Real-Time Rendering and Essential Mathematics for Games

    float tMin = 0.0f;
    float tMax = 100000.0f;

    glm::vec3 OBBposition_worldspace(ModelMatrix[3].x, ModelMatrix[3].y, ModelMatrix[3].z);

    glm::vec3 delta = OBBposition_worldspace - ray_origin;

    // Test intersection with the 2 planes perpendicular to the OBB's X axis
    {
        glm::vec3 xaxis(ModelMatrix[0].x, ModelMatrix[0].y, ModelMatrix[0].z);
        float e = glm::dot(xaxis, delta);
        float f = glm::dot(ray_direction, xaxis);

        if (fabs(f) > 0.001f) { // Standard case

            float t1 = (e + aabb_min.x) / f; // Intersection with the "left" plane
            float t2 = (e + aabb_max.x) / f; // Intersection with the "right" plane
            // t1 and t2 now contain distances between ray origin and ray-plane intersections

            // We want t1 to represent the nearest intersection,
            // so if it's not the case, invert t1 and t2
            if (t1 > t2) {
                float w = t1;
                t1 = t2;
                t2 = w; // swap t1 and t2
            }

            // tMax is the nearest "far" intersection (amongst the X,Y and Z planes pairs)
            if (t2 < tMax)
                tMax = t2;
            // tMin is the farthest "near" intersection (amongst the X,Y and Z planes pairs)
            if (t1 > tMin)
                tMin = t1;

            // And here's the trick :
            // If "far" is closer than "near", then there is NO intersection.
            // See the images in the tutorials for the visual explanation.
            if (tMax < tMin)
                return false;

        } else { // Rare case : the ray is almost parallel to the planes, so they don't have any "intersection"
            if (-e + aabb_min.x > 0.0f || -e + aabb_max.x < 0.0f)
                return false;
        }
    }

    // Test intersection with the 2 planes perpendicular to the OBB's Y axis
    // Exactly the same thing than above.
    {
        glm::vec3 yaxis(ModelMatrix[1].x, ModelMatrix[1].y, ModelMatrix[1].z);
        float e = glm::dot(yaxis, delta);
        float f = glm::dot(ray_direction, yaxis);

        if (fabs(f) > 0.001f) {

            float t1 = (e + aabb_min.y) / f;
            float t2 = (e + aabb_max.y) / f;

            if (t1 > t2) {
                float w = t1;
                t1 = t2;
                t2 = w;
            }

            if (t2 < tMax)
                tMax = t2;
            if (t1 > tMin)
                tMin = t1;
            if (tMin > tMax)
                return false;

        } else {
            if (-e + aabb_min.y > 0.0f || -e + aabb_max.y < 0.0f)
                return false;
        }
    }

    // Test intersection with the 2 planes perpendicular to the OBB's Z axis
    // Exactly the same thing than above.
    {
        glm::vec3 zaxis(ModelMatrix[2].x, ModelMatrix[2].y, ModelMatrix[2].z);
        float e = glm::dot(zaxis, delta);
        float f = glm::dot(ray_direction, zaxis);

        if (fabs(f) > 0.001f) {

            float t1 = (e + aabb_min.z) / f;
            float t2 = (e + aabb_max.z) / f;

            if (t1 > t2) {
                float w = t1;
                t1 = t2;
                t2 = w;
            }

            if (t2 < tMax)
                tMax = t2;
            if (t1 > tMin)
                tMin = t1;
            if (tMin > tMax)
                return false;

        } else {
            if (-e + aabb_min.z > 0.0f || -e + aabb_max.z < 0.0f)
                return false;
        }
    }

    intersection_distance = tMin;
    return true;
}

void print_vec(const char* description, glm::vec3 vec) {
    printf("%s: %.2lf, %.2lf, %.2lf\n", description, vec[0], vec[1], vec[2]);
}

bool rayIntersectRing(glm::vec3 ray_origin, glm::vec3 ray_direction, glm::vec3 center, float o_radius, float i_radius,
                      glm::vec3 a, glm::vec3 b, float thickness, glm::mat4 ModelMatrix) {
    // first test an obb
    float t;
    auto diagonal = glm::normalize(a + b);
    auto normal = glm::normalize(glm::cross(a, b));
    auto aabb_max = center + thickness * normal + o_radius * diagonal;
    auto aabb_min = center - thickness * normal - o_radius * diagonal;
//    if (!rayIntersectOBB(ray_origin, ray_direction, aabb_min, aabb_max, ModelMatrix, t)) {
//        return false;
//    }

    // next test ring
    auto tc = ModelMatrix * glm::vec4(center, 1.0);
    center = tc / tc[3];
    t = glm::dot(center - ray_origin, normal) / glm::dot(ray_direction, normal);
    auto p = ray_origin + t * ray_direction;
    if (t < 0) return false;
    auto distance = glm::length(p - center);
    return distance > i_radius && distance < o_radius;
}

std::optional<float> rayIntersectSphere(glm::vec3 ray_origin, glm::vec3 ray_direction, glm::vec3 center, float radius) {
    auto &p = ray_origin;
    auto &d = ray_direction;
    auto &c = center;
    auto &r = radius;
    glm::vec3 L = c - p;
    float tca = glm::dot(L, d);
    if (tca < 0) return std::nullopt;
    float d2 = glm::dot(L, L) - tca * tca;
    float radius2 = radius * radius;
    if (d2 > radius2) return std::nullopt;
    float thc = sqrt(radius2 - d2);
    return tca - thc;
}

}
}