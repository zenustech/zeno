#pragma once

#if !defined(__CUDACC_RTC__)
#include <cuda_runtime.h>
#endif
#include <cuda/cstdint.h>
#include <sutil/vec_math.h>

struct HitGroupData;

namespace Mnee {

static constexpr int MNEE_MAX_ITERATIONS = 64;
static constexpr int MNEE_MAX_INTERSECTION_COUNT = 10;
static constexpr int MNEE_MAX_CAUSTIC_CASTERS = 6;
static constexpr float MNEE_SOLVER_THRESHOLD = 0.001f;
static constexpr float MNEE_MINIMUM_STEP_SIZE = 0.0001f;
static constexpr float MNEE_MIN_DISTANCE = 0.001f;
static constexpr float MNEE_MIN_PROGRESS_DISTANCE = 0.0001f;
static constexpr float MNEE_MIN_DETERMINANT = 0.0001f;
static constexpr float MNEE_PROJECTION_DISTANCE_MULTIPLIER = 2.0f;
// CallableDefault clamps even a zero-roughness shader to 0.01. Treat that
// renderer minimum as smooth glass for this (delta transmission) solver.
static constexpr float SmoothRoughnessLimit = 0.010001f;

struct ManifoldMaterial {
    float eta = 1.0f;
    float specular = 1.0f;
    float f0 = 0.04f;
    float glassWeight = 0.0f;
    float3 transmissionColor = make_float3(1.0f);
    float3 extinction = {};
    const HitGroupData* sourceSbt = nullptr;
};

struct ManifoldGeometry {
    // p, dp_du/dp_dv and dn_du/dn_dv, in the renderer's camera-relative world space.
    float3 p = {};
    float3 dp_du = {};
    float3 dp_dv = {};
    float3 n = {};
    float3 ng = {};
    float3 dn_du = {};
    float3 dn_dv = {};

    uint32_t instanceId = 0;
};

struct ManifoldVertex : ManifoldGeometry {
    ManifoldMaterial material;

    float2 constraint = {};
    // Row-major 2x2 blocks: a = dC_i/dx_(i-1), b = dC_i/dx_i,
    // c = dC_i/dx_(i+1). x_i is a displacement in (dp_du, dp_dv).
    float4 a = {};
    float4 b = make_float4(1.0f, 0.0f, 0.0f, 1.0f);
    float4 c = {};
};

static __forceinline__ __host__ __device__ bool normalizeWithLength(float3& v, float& lengthOut, bool inverse)
{
    lengthOut = length(v);
    if ((lengthOut <= MNEE_MIN_DISTANCE) || !isfinite(lengthOut)) {
        return false;
    }
    auto tmp = 1.0f / lengthOut;
    if (inverse)
        lengthOut = tmp;
    v *= tmp;
    return true;
}

static __forceinline__ __host__ __device__
void makeOrthonormals(const float3& n, float3& tangent, float3& bitangent)
{
    if (fabsf(n.x) > fabsf(n.z)) {
        tangent = normalize(make_float3(-n.y, n.x, 0.0f));
    }
    else {
        tangent = normalize(make_float3(0.0f, -n.z, n.y));
    }
    bitangent = cross(n, tangent);
}

// Mnee_setup_manifold_vertex: geometry-only portion, after object transforms.
// Inputs are triangle vertices/normals in the same space; barys = (u,v).
// The sign flag preserves the outward normal under mirrored instance transforms.
static __forceinline__ __host__ __device__ bool mnee_setup_manifold_geometry(
    const float3 worldVertices[3], const float3 worldNormals[3], const float2& barys, 
    ManifoldGeometry& vertex, bool negativeScale = false)
{
    if (!isfinite(barys.x) || !isfinite(barys.y)) {
        return false;
    }
    const float bary0 = 1.0f - barys.x - barys.y;
    vertex.p = bary0 * worldVertices[0] + barys.x * worldVertices[1] + barys.y * worldVertices[2];

    float3 dp_du = worldVertices[1] - worldVertices[0];
    float3 dp_dv = worldVertices[2] - worldVertices[0];
    const float3 geometricCross = cross(dp_du, dp_dv);
    const float geometricLength = length(geometricCross);
    // Triangle area has squared-distance units. A ray-distance threshold here
    // incorrectly rejects small, otherwise well-conditioned triangles.
    if (!(geometricLength > 0.0f) || !isfinite(geometricLength)) {
        return false;
    }
    vertex.ng = geometricCross / geometricLength;
    if (negativeScale) {
        vertex.ng = -vertex.ng;
    }

    const float3 unnormalizedNormal = bary0 * worldNormals[0] + barys.x * worldNormals[1] + barys.y * worldNormals[2];
    const float normalLength = length(unnormalizedNormal);
    if (!(normalLength > 0.0f) || !isfinite(normalLength)) {
        return false;
    }
    vertex.n = unnormalizedNormal / normalLength;

    // Differentiate normalize(N(u,v)): project each derivative onto n's tangent
    // plane, then divide by the length of the unnormalized interpolated normal.
    const float inverseNormalLength = 1.0f / normalLength;
    float3 dn_du = inverseNormalLength * (worldNormals[1] - worldNormals[0]);
    float3 dn_dv = inverseNormalLength * (worldNormals[2] - worldNormals[0]);
    dn_du -= vertex.n * dot(vertex.n, dn_du);
    dn_dv -= vertex.n * dot(vertex.n, dn_dv);

    const float dpduLength = length(dp_du);
    if (!(dpduLength > 0.0f) || !isfinite(dpduLength)) {
        return false;
    }
    dp_du /= dpduLength;
    dn_du /= dpduLength;

    // Apply the same Gram-Schmidt change of coordinates to position and normal
    // derivatives; otherwise the Newton Jacobian mixes incompatible UV frames.
    const float dpduDotDpdv = dot(dp_du, dp_dv);
    dp_dv -= dpduDotDpdv * dp_du;
    dn_dv -= dpduDotDpdv * dn_du;
    const float dpdvLength = length(dp_dv);
    if (!(dpdvLength > 0.0f) || !isfinite(dpdvLength)) {
        return false;
    }
    dp_dv /= dpdvLength;
    dn_dv /= dpdvLength;

    makeOrthonormals(vertex.ng, vertex.dp_du, vertex.dp_dv);
    // Use the full orthogonal change of basis so mirrored instances
    // (which also reverse handedness) are handled correctly.
    vertex.dn_du = dot(vertex.dp_du, dp_du) * dn_du + dot(vertex.dp_du, dp_dv) * dn_dv;
    vertex.dn_dv = dot(vertex.dp_dv, dp_du) * dn_du + dot(vertex.dp_dv, dp_dv) * dn_dv;

    return isfinite(vertex.p.x) && isfinite(vertex.p.y) && isfinite(vertex.p.z) &&
           isfinite(vertex.dn_du.x) && isfinite(vertex.dn_du.y) && isfinite(vertex.dn_du.z) &&
           isfinite(vertex.dn_dv.x) && isfinite(vertex.dn_dv.y) && isfinite(vertex.dn_dv.z);
}

static __forceinline__ __host__ __device__ float4 mat22_mult(const float4& a, const float4& b)
{
    return make_float4(a.x * b.x + a.y * b.z,
                       a.x * b.y + a.y * b.w,
                       a.z * b.x + a.w * b.z,
                       a.z * b.y + a.w * b.w);
}

static __forceinline__ __host__ __device__ float2 mat22_mult(const float4& a, const float2& b)
{
    return make_float2(a.x * b.x + a.y * b.y, a.z * b.x + a.w * b.y);
}

static __forceinline__ __host__ __device__ float mat22_determinant(const float4& matrix)
{
    return matrix.x * matrix.w - matrix.y * matrix.z;
}

static __forceinline__ __host__ __device__ float mat22_inverse(const float4& matrix, float4& inverse)
{
    // Constraint derivatives have inverse-length units. An absolute determinant
    // cutoff rejects well-conditioned paths when the scene is scaled up.
    const float scale = fmaxf(fmaxf(fabsf(matrix.x), fabsf(matrix.y)),
                              fmaxf(fabsf(matrix.z), fabsf(matrix.w)));
    if (!(scale > 0.0f) || !isfinite(scale)) {
        return 0.0f;
    }
    const float4 scaled = matrix / scale;
    const float determinant = mat22_determinant(scaled);
    if (fabsf(determinant) < MNEE_MIN_DETERMINANT || !isfinite(determinant)) {
        return 0.0f;
    }
    // Preserve return contract: det(matrix), not det(matrix / scale).
    // Reject determinants that cannot be represented by this float interface.
    const float originalDeterminant = static_cast<float>(static_cast<double>(determinant) * scale * scale);
    if (originalDeterminant == 0.0f || !isfinite(originalDeterminant)) {
        return 0.0f;
    }
    inverse = make_float4(scaled.w, -scaled.y, -scaled.z, scaled.x) / (determinant * scale);
    return originalDeterminant;
}

static __forceinline__ __host__ __device__ float effectiveEta(const ManifoldVertex& vertex, const float3& wi)
{
    return dot(wi, vertex.ng) < 0.0f ? 1.0f / vertex.material.eta : vertex.material.eta;
}

static __forceinline__ __host__ __device__ bool mnee_compute_constraint_derivatives(
    int vertexCount,
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS],
    const float3& receiverPosition,
    bool lightFixedDirection,
    const float3& lightSample)
{
    if (vertexCount < 1 || vertexCount > MNEE_MAX_CAUSTIC_CASTERS) {
        return false;
    }
    // C_i = (dot(s_i,H_i), dot(t_i,H_i)), H_i = normalize(-(wi + eta*wo)).
    // For smooth refraction C_i=0 means Snell's law is satisfied. Each C_i
    // depends only on the previous/current/next vertex, giving three blocks.
    for (int vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex) {
        ManifoldVertex& vertex = vertices[vertexIndex];
        vertex.a = vertex.c = {};

        float3 wi = (vertexIndex == 0 ? receiverPosition: vertices[vertexIndex - 1].p) - vertex.p;
        float wiInverseLength;
        if (!normalizeWithLength(wi, wiInverseLength, true)) {
            return false;
        }

        float3 wo = (vertexIndex == vertexCount - 1)
                        ? (lightFixedDirection ? lightSample : lightSample - vertex.p)
                        : vertices[vertexIndex + 1].p - vertex.p;
        float woInverseLength;
        if (!normalizeWithLength(wo, woInverseLength, true)) {
            return false;
        }

        const float eta = effectiveEta(vertex, wi);
        float3 halfVector = -(wi + eta * wo);
        const float halfLength = length(halfVector);
        if (!(halfLength > MNEE_MIN_DISTANCE)) {
            return false;
        }
        const float inverseHalfLength = 1.0f / halfLength;
        halfVector *= inverseHalfLength;
        woInverseLength *= eta * inverseHalfLength;
        wiInverseLength *= inverseHalfLength;

        const float dpduDotNormal = dot(vertex.dp_du, vertex.n);
        float3 tangent = vertex.dp_du - dpduDotNormal * vertex.n;
        const float tangentLength = length(tangent);
        if (!(tangentLength > MNEE_MIN_DISTANCE)) {
            return false;
        }
        const float inverseTangentLength = 1.0f / tangentLength;
        tangent *= inverseTangentLength;
        const float3 bitangent = cross(vertex.n, tangent);

        float3 dHalfDu;
        float3 dHalfDv;
        // a: change the incoming direction by moving the previous vertex.
        if (vertexIndex > 0) {
            const ManifoldVertex& previous = vertices[vertexIndex - 1];
            dHalfDu = (previous.dp_du - wi * dot(wi, previous.dp_du)) * wiInverseLength;
            dHalfDv = (previous.dp_dv - wi * dot(wi, previous.dp_dv)) * wiInverseLength;
            dHalfDu -= halfVector * dot(dHalfDu, halfVector);
            dHalfDv -= halfVector * dot(dHalfDv, halfVector);
            dHalfDu = -dHalfDu;
            dHalfDv = -dHalfDv;
            vertex.a = make_float4(dot(dHalfDu, tangent), dot(dHalfDv, tangent),
                                   dot(dHalfDu, bitangent), dot(dHalfDv, bitangent));
        }

        // b: moving this vertex changes both directions, except that the last
        // outgoing direction is constant for a directional/environment sample.
        if (vertexIndex == vertexCount - 1 && lightFixedDirection) {
            dHalfDu = wiInverseLength *
                      (-vertex.dp_du + wi * dot(wi, vertex.dp_du));
            dHalfDv = wiInverseLength *
                      (-vertex.dp_dv + wi * dot(wi, vertex.dp_dv));
        }
        else {
            dHalfDu = -vertex.dp_du * (wiInverseLength + woInverseLength) +
                      wi * (dot(wi, vertex.dp_du) * wiInverseLength) +
                      wo * (dot(wo, vertex.dp_du) * woInverseLength);
            dHalfDv = -vertex.dp_dv * (wiInverseLength + woInverseLength) +
                      wi * (dot(wi, vertex.dp_dv) * wiInverseLength) +
                      wo * (dot(wo, vertex.dp_dv) * woInverseLength);
        }
        dHalfDu -= halfVector * dot(dHalfDu, halfVector);
        dHalfDv -= halfVector * dot(dHalfDv, halfVector);
        dHalfDu = -dHalfDu;
        dHalfDv = -dHalfDv;

        // The local shading frame also moves with the shading normal. These
        // terms are essential on smooth/curved surfaces, even with zero roughness.
        float3 dTangentDu = -inverseTangentLength *
            (dot(vertex.dp_du, vertex.dn_du) * vertex.n + dpduDotNormal * vertex.dn_du);
        float3 dTangentDv = -inverseTangentLength *
            (dot(vertex.dp_du, vertex.dn_dv) * vertex.n + dpduDotNormal * vertex.dn_dv);
        dTangentDu -= tangent * dot(tangent, dTangentDu);
        dTangentDv -= tangent * dot(tangent, dTangentDv);
        const float3 dBitangentDu = cross(vertex.dn_du, tangent) +
                                    cross(vertex.n, dTangentDu);
        const float3 dBitangentDv = cross(vertex.dn_dv, tangent) +
                                    cross(vertex.n, dTangentDv);

        vertex.b = make_float4(dot(dHalfDu, tangent) + dot(halfVector, dTangentDu),
                               dot(dHalfDv, tangent) + dot(halfVector, dTangentDv),
                               dot(dHalfDu, bitangent) + dot(halfVector, dBitangentDu),
                               dot(dHalfDv, bitangent) + dot(halfVector, dBitangentDv));

        // c: change the outgoing direction by moving the next vertex.
        if (vertexIndex < vertexCount - 1) {
            const ManifoldVertex& next = vertices[vertexIndex + 1];
            dHalfDu = (next.dp_du - wo * dot(wo, next.dp_du)) * woInverseLength;
            dHalfDv = (next.dp_dv - wo * dot(wo, next.dp_dv)) * woInverseLength;
            dHalfDu -= halfVector * dot(dHalfDu, halfVector);
            dHalfDv -= halfVector * dot(dHalfDv, halfVector);
            dHalfDu = -dHalfDu;
            dHalfDv = -dHalfDv;
            vertex.c = make_float4(dot(dHalfDu, tangent), dot(dHalfDv, tangent),
                                   dot(dHalfDu, bitangent), dot(dHalfDv, bitangent));
        }

        vertex.constraint = make_float2(dot(tangent, halfVector), dot(bitangent, halfVector));
    }
    return true;
}

static __forceinline__ __host__ __device__
bool mnee_solve_matrix_h_to_x(int vertexCount, ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS], float2 dx[MNEE_MAX_CAUSTIC_CASTERS])
{
    if (vertexCount < 1 || vertexCount > MNEE_MAX_CAUSTIC_CASTERS) {
        return false;
    }
    // Block Thomas/LU solve J*dx=C. Newton subsequently subtracts dx. Eliminate
    // the lower blocks forward, then back-substitute; matrix order matters.
    float4 inverse[MNEE_MAX_CAUSTIC_CASTERS];
    float2 constraint[MNEE_MAX_CAUSTIC_CASTERS];

    float4 diagonal = vertices[0].b;
    if (mat22_inverse(diagonal, inverse[0]) == 0.0f) {
        return false;
    }
    constraint[0] = vertices[0].constraint;

    for (int i = 1; i < vertexCount; ++i) {
        const float4 lower = mat22_mult(vertices[i].a, inverse[i - 1]);
        diagonal = vertices[i].b - mat22_mult(lower, vertices[i - 1].c);
        if (mat22_inverse(diagonal, inverse[i]) == 0.0f) {
            return false;
        }
        constraint[i] = vertices[i].constraint - mat22_mult(lower, constraint[i - 1]);
    }

    dx[vertexCount - 1] = mat22_mult(inverse[vertexCount - 1], constraint[vertexCount - 1]);
    for (int i = vertexCount - 2; i >= 0; --i) {
        dx[i] = mat22_mult(
            inverse[i], constraint[i] - mat22_mult(vertices[i].c, dx[i + 1]));
    }
    return true;
}

template <typename LightSample, typename Project>
static __forceinline__ __host__ __device__ bool mnee_newton_solver(
    const float3& receiverPosition,
    const float3& receiverNormal,
    const LightSample& lightSample,
    bool lightFixedDirection,
    uint32_t seed,
    int vertexCount,
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS],
    const Project& project)
{
    if (vertexCount < 1 || vertexCount > MNEE_MAX_CAUSTIC_CASTERS) {
        return false;
    }
    float2 dx[MNEE_MAX_CAUSTIC_CASTERS];
    // Tentative steps only change geometry. Keep immutable seed closures and
    // current constraint matrices out of this repeatedly copied scratch array.
    ManifoldGeometry tentative[MNEE_MAX_CAUSTIC_CASTERS];
    const float3 lightTarget = lightFixedDirection ? lightSample.dir : lightSample.p;

    float beta = 0.1f;
    // Start conservatively, project the tangent-plane proposal back onto the
    // same caster, and halve beta if projection/transmission checks fail.
    // Project is the OptiX adapter in production, an analytic surface in tests.
    bool reduceStep = false;
    bool resolveConstraint = true;
    for (int iteration = 0; iteration < MNEE_MAX_ITERATIONS; ++iteration) {
        if (resolveConstraint) {
            if (!mnee_compute_constraint_derivatives(
                    vertexCount, vertices, receiverPosition, lightFixedDirection, lightTarget))
            {
                return false;
            }

            float constraintNorm = 0.0f;
            for (int i = 0; i < vertexCount; ++i) {
                constraintNorm = fmaxf(constraintNorm, length(vertices[i].constraint));
            }
            if (constraintNorm < MNEE_SOLVER_THRESHOLD) {
                return true;
            }
            if (!mnee_solve_matrix_h_to_x(vertexCount, vertices, dx)) {
                return false;
            }
        }

        for (int i = 0; i < vertexCount; ++i) {
            const ManifoldVertex& vertex = vertices[i];
            float3 target = vertex.p - beta *
                (dx[i].x * vertex.dp_du + dx[i].y * vertex.dp_dv);
            if (i == 0 && dot(target - receiverPosition, receiverNormal) <= 0.0f) {
                target = vertex.p + beta *
                    (dx[i].x * vertex.dp_du + dx[i].y * vertex.dp_dv);
            }

            const float3 projectionOrigin = i == 0 ? receiverPosition : vertices[i - 1].p;
            if (!project(
                    projectionOrigin, target, vertex.instanceId, vertex.material, seed, tentative[i]))
            {
                reduceStep = true;
                break;
            }
            if (length(tentative[i].p - vertex.p) < MNEE_MIN_PROGRESS_DISTANCE) {
                return false;
            }
        }

        if (!reduceStep) {
            for (int i = 0; i < vertexCount; ++i) {
                const ManifoldGeometry& vertex = tentative[i];
                const float3 wi = (i == 0 ? receiverPosition : tentative[i - 1].p) - vertex.p;
                const float3 wo = i == vertexCount - 1
                                      ? (lightFixedDirection ? lightSample.dir
                                                             : lightSample.p - vertex.p)
                                      : tentative[i + 1].p - vertex.p;
                if (dot(vertex.n, wi) * dot(vertex.n, wo) >= 0.0f) {
                    reduceStep = true;
                    break;
                }
            }
        }

        if (reduceStep) {
            reduceStep = false;
            resolveConstraint = false;
            beta *= 0.5f;
            if (beta < MNEE_MINIMUM_STEP_SIZE) {
                return false;
            }
            continue;
        }

        for (int i = 0; i < vertexCount; ++i) {
            static_cast<ManifoldGeometry&>(vertices[i]) = tentative[i];
        }
        resolveConstraint = true;
        beta = fminf(1.0f, 2.0f * beta);
    }
    return false;
}

template <typename LightSample>
static __forceinline__ __host__ __device__ bool mnee_compute_transfer_matrix(
    const float3& receiverPosition,
    const LightSample& lightSample,
    bool lightFixedDirection,
    int vertexCount,
    ManifoldVertex vertices[MNEE_MAX_CAUSTIC_CASTERS],
    float& firstVertexToLight)
{
    if (vertexCount < 1 || vertexCount > MNEE_MAX_CAUSTIC_CASTERS) {
        return false;
    }
    // Solve J * dx/dlight = -dC/dlight and propagate from the last interface
    // back to the first. Its absolute determinant converts the light's sample
    // measure into area at the first caster (solid angle for fixed directions,
    // emitter area otherwise). This is not the receiver BSDF or its cosine.
    float4 inverse;
    float4 upper[MNEE_MAX_CAUSTIC_CASTERS - 1];
    float4 diagonal = vertices[0].b;
    if (mat22_inverse(diagonal, inverse) == 0.0f) {
        return false;
    }

    for (int i = 1; i < vertexCount; ++i) {
        upper[i - 1] = mat22_mult(inverse, vertices[i - 1].c);
        diagonal = vertices[i].b - mat22_mult(vertices[i].a, upper[i - 1]);
        if (mat22_inverse(diagonal, inverse) == 0.0f) {
            return false;
        }
    }

    const int lastIndex = vertexCount - 1;
    const ManifoldVertex& last = vertices[lastIndex];
    const float3 tangent = normalize(last.dp_du - dot(last.dp_du, last.n) * last.n);
    const float3 bitangent = cross(last.n, tangent);

    float3 wi = (vertexCount == 1 ? receiverPosition : vertices[lastIndex - 1].p) - last.p;
    float wiLength;
    if (!normalizeWithLength(wi, wiLength, false)) {
        return false;
    }
    const float eta = effectiveEta(last, wi);

    float endpointMeasure = 1.0f;
    float4 endpointDerivative;
    if (lightFixedDirection) {
        const float3 wo = lightSample.dir;
        float3 halfVector = -(wi + eta * wo);
        const float halfLength = length(halfVector);
        if (!(halfLength > MNEE_MIN_DISTANCE)) {
            return false;
        }
        halfVector /= halfLength;
        const float inverseWoLength = -eta / halfLength;

        // Parameterize direction in its tangent plane: d(omega) = du dv at
        // the sample. Spherical coordinates collapse at normal incidence,
        // making the old determinant zero for an overhead directional light.
        float3 lightDu, lightDv;
        makeOrthonormals(wo, lightDu, lightDv);
        float3 dHalfDu = inverseWoLength * lightDu;
        float3 dHalfDv = inverseWoLength * lightDv;
        dHalfDu -= halfVector * dot(dHalfDu, halfVector);
        dHalfDv -= halfVector * dot(dHalfDv, halfVector);
        endpointDerivative = make_float4(dot(dHalfDu, tangent), dot(dHalfDv, tangent),
                                        dot(dHalfDu, bitangent), dot(dHalfDv, bitangent));
    }
    else {
        float3 lightDu, lightDv;
        makeOrthonormals(normalize(lightSample.n), lightDu, lightDv);
        float3 wo = lightSample.p - last.p;
        float inverseWoLength;
        if (!normalizeWithLength(wo, inverseWoLength, true)) {
            return false;
        }

        float3 halfVector = -(wi + eta * wo);
        const float halfLength = length(halfVector);
        if (!(halfLength > MNEE_MIN_DISTANCE)) {
            return false;
        }
        halfVector /= halfLength;
        inverseWoLength *= eta / halfLength;

        float3 dHalfDu = (lightDu - wo * dot(wo, lightDu)) * inverseWoLength;
        float3 dHalfDv = (lightDv - wo * dot(wo, lightDv)) * inverseWoLength;
        dHalfDu -= halfVector * dot(dHalfDu, halfVector);
        dHalfDv -= halfVector * dot(dHalfDv, halfVector);
        dHalfDu = -dHalfDu;
        dHalfDv = -dHalfDv;
        endpointDerivative = make_float4(dot(dHalfDu, tangent), dot(dHalfDv, tangent),
                                        dot(dHalfDu, bitangent), dot(dHalfDv, bitangent));
    }

    float4 transfer = -mat22_mult(inverse, endpointDerivative);
    for (int i = vertexCount - 2; i >= 0; --i) {
        transfer = -mat22_mult(upper[i], transfer);
    }
    firstVertexToLight = fabsf(mat22_determinant(transfer)) * endpointMeasure;
    return isfinite(firstVertexToLight);
}

} // namespace Mnee
