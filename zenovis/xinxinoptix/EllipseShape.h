#pragma once

#include <Sampling.h>
#include <SphericalEllipse.h>

// Sampling view of an ellipse in its original light-plane coordinates.
// RectShape retains the shared host/device storage; this value view keeps all
// ellipse proposal construction, sampling, UV recovery, and PDFs separate.
struct EllipseShape {
    float3 v;
    float3 axisX; float lenX;
    float3 axisY; float lenY;
    float3 normal;

    float3 center() const {
        return v + axisX * 0.5f * lenX + axisY * 0.5f * lenY;
    }

    inline float AreaPdf(const float3& shadingP, float dist) const {
        // cos(theta) = planeDistance / dist; this form stays accurate for
        // nearly tangent rays, where dot(dir, normal) loses precision.
        const float planeDistance = fabsf(dot(center() - shadingP, normal));
        const float ellipseArea = 0.25f * M_PIf * lenX * lenY;
        const float pdf = dist * dist * dist / (planeDistance * ellipseArea);
        return planeDistance > 0.0f && ellipseArea > 0.0f && isfinite(pdf) ? pdf : 0.0f;
    }

    inline float PdfAsLight(const float3& shadingP, float dist) const {
        SphericalEllipse ellipse;
        if (ellipse.init(center() - shadingP, axisX, 0.5f * lenX, axisY, 0.5f * lenY))
            return 1.0f / ellipse.solidAngle;
        return AreaPdf(shadingP, dist);
    }

    struct Proposal {
        enum Mode { Empty, FullEllipse, SpreadCone, CroppedArea };
        Mode mode = Empty;
        float4 bounds; // minU, maxU, minV, maxV in the original light plane
        float coneHeight, solidAngle, planeDistance, spreadRadius;
        float2 receiver;
        SphericalEllipse ellipse;
        bool spherical = false;
    };

    inline Proposal ProposalFor(const float3& shadingP, float spread) const {
        Proposal p{};
        if (!(lenX > 0.0f && lenY > 0.0f && spread > 0.0f)) return p;
        p.planeDistance = fabsf(dot(center() - shadingP, normal));
        if (!(p.planeDistance > 0.0f) || !isfinite(p.planeDistance)) return p;
        p.receiver = make_float2(dot(shadingP - v, axisX), dot(shadingP - v, axisY));
        p.mode = Proposal::FullEllipse;
        if (spread >= 1.0f) {
            p.spherical = p.ellipse.init(center() - shadingP, axisX, .5f*lenX, axisY, .5f*lenY);
            return p;
        }

        // World-to-light projections lose a few ulps for small, distant lights.
        // Enlarge only the proposal, never the emitter or physical attenuation.
        const float anglePad = 8.0f*__FLT_EPSILON__*fmaxf(1.0f,length(center()-shadingP)/p.planeDistance);
        const float angle = fminf(1.5707962513f,.5f*M_PIf*spread+anglePad);
        const float radius = p.planeDistance * tanf(angle);
        p.spreadRadius=radius;
        p.bounds = make_float4(fmaxf(0.0f, p.receiver.x-radius), fminf(lenX, p.receiver.x+radius),
                               fmaxf(0.0f, p.receiver.y-radius), fminf(lenY, p.receiver.y+radius));
        if (!(p.bounds.y > p.bounds.x && p.bounds.w > p.bounds.z)) { p.mode=Proposal::Empty; return p; }
        // The closest point of this rectangle to the ellipse center gives an
        // exact rectangle/ellipse overlap test. Never discard a contributing sliver.
        const float closestX = clamp(.5f*lenX,p.bounds.x,p.bounds.y)/lenX-.5f;
        const float closestY = clamp(.5f*lenY,p.bounds.z,p.bounds.w)/lenY-.5f;
        if (closestX*closestX+closestY*closestY >= .25f) { p.mode=Proposal::Empty; return p; }

        // 1-cos(angle), evaluated without cancellation for very small spreads.
        const float sine = sinf(.5f*angle);
        p.coneHeight = 2.0f*sine*sine;
        p.solidAngle = 2.0f*M_PIf*p.coneHeight;
        if (!(p.solidAngle > 0.0f)) { p.mode=Proposal::Empty; return p; }
        p.mode = Proposal::SpreadCone;

        const float nearX = p.receiver.x-clamp(p.receiver.x,p.bounds.x,p.bounds.y);
        const float nearY = p.receiver.y-clamp(p.receiver.y,p.bounds.z,p.bounds.w);
        const float near2 = p.planeDistance*p.planeDistance+nearX*nearX+nearY*nearY;
        const float cropArea = (p.bounds.y-p.bounds.x)*(p.bounds.w-p.bounds.z);
        // Upper bound on the rectangle's solid angle. Uniform area sampling
        // avoids the nearly degenerate spherical-rectangle inversion at edges.
        const float cropBound = cropArea*p.planeDistance/(near2*sqrtf(near2));
        float score = p.solidAngle;
        if (cropBound > 0.0f && cropBound < score) { p.mode=Proposal::CroppedArea; score=cropBound; }

        const float farX = fmaxf(fabsf(p.receiver.x),fabsf(lenX-p.receiver.x));
        const float farY = fmaxf(fabsf(p.receiver.y),fabsf(lenY-p.receiver.y));
        const float far2 = p.planeDistance*p.planeDistance+farX*farX+farY*farY;
        const float ellipseLower = (.25f*M_PIf*lenX*lenY)*p.planeDistance/(far2*sqrtf(far2));
        // Skip eigendecomposition when a conservative bound already proves the
        // cone/crop is smaller. Proposal selection is deterministic at the receiver.
        if (score < ellipseLower) return p;
        p.spherical = p.ellipse.init(center() - shadingP,axisX,.5f*lenX,axisY,.5f*lenY);
        const float ex = p.receiver.x-clamp(p.receiver.x,0.0f,lenX);
        const float ey = p.receiver.y-clamp(p.receiver.y,0.0f,lenY);
        const float nearEllipse2 = p.planeDistance*p.planeDistance+ex*ex+ey*ey;
        const float ellipseScore = p.spherical ? p.ellipse.solidAngle :
            (.25f*M_PIf*lenX*lenY)*p.planeDistance/(nearEllipse2*sqrtf(nearEllipse2));
        if (ellipseScore < score) p.mode=Proposal::FullEllipse;
        return p;
    }

    inline bool SpreadContains(const Proposal& p, const float2& uv, float spread) const {
        const float2 edge = uv-.5f;
        // Small allowance covers reintersection/UV rounding on the ellipse rim.
        if (dot(edge,edge) > .2500005f) return false;
        if (spread >= 1.0f) return true;
        const float du = uv.x*lenX-p.receiver.x, dv = uv.y*lenY-p.receiver.y;
        const float radius = p.spreadRadius;
        return du*du+dv*dv <= radius*radius*(1.0f+2e-6f);
    }

    inline float ProposalPdf(const Proposal& p, float dist) const {
        if (p.mode==Proposal::Empty) return 0.0f;
        if (p.mode==Proposal::SpreadCone) return 1.0f/p.solidAngle;
        if (p.mode==Proposal::FullEllipse && p.spherical) return 1.0f/p.ellipse.solidAngle;
        const float proposalArea = p.mode==Proposal::CroppedArea ?
            (p.bounds.y-p.bounds.x)*(p.bounds.w-p.bounds.z) : .25f*M_PIf*lenX*lenY;
        const float pdf = dist*dist*dist/(p.planeDistance*proposalArea);
        return pdf > 0.0f && isfinite(pdf) ? pdf : 0.0f;
    }

    inline float SpreadPdfAsLight(const float3& shadingP, const float2& uv, float dist, float spread) const {
        const Proposal p=ProposalFor(shadingP,spread);
        if (p.mode==Proposal::Empty || !SpreadContains(p,uv,spread)) return 0.0f;
        return ProposalPdf(p,dist);
    }

    inline void SampleWithSpread(LightSampleRecord* lsr, const float2& u, const float3& shadingP, float spread,
                                 bool useLut = ZENO_ELLIPSE_USE_LUT != 0) const {
        lsr->PDF=0.0f;
        const Proposal p=ProposalFor(shadingP,spread);
        if (p.mode==Proposal::Empty) return;
        if (p.mode==Proposal::SpreadCone || (p.mode==Proposal::FullEllipse && p.spherical)) {
            const float signedDistance = dot(center()-shadingP,normal);
            if (p.mode==Proposal::SpreadCone) {
                const float height = clamp(u.x,0.0f,1.0f)*p.coneHeight;
                const float radial = sqrtf(fmaxf(0.0f,height*(2.0f-height)));
                const float phi = 2.0f*M_PIf*u.y;
                const float3 pole = signedDistance < 0.0f ? -normal : normal;
                lsr->dir = normalize(pole*(1.0f-height)+radial*(axisX*cosf(phi)+axisY*sinf(phi)));
            } else lsr->dir=p.ellipse.sampleWithMethod(u,useLut);
            lsr->dist = signedDistance/dot(lsr->dir,normal);
            lsr->p = shadingP+lsr->dist*lsr->dir;
            const float3 delta = lsr->p-v;
            lsr->uv = make_float2(dot(delta,axisX)/lenX,dot(delta,axisY)/lenY);
        } else {
            float pu, pv;
            if (p.mode==Proposal::CroppedArea) {
                pu = p.bounds.x+(p.bounds.y-p.bounds.x)*u.x;
                pv = p.bounds.z+(p.bounds.w-p.bounds.z)*u.y;
            } else {
                const float radius=sqrtf(u.x), phi=2.0f*M_PIf*u.y;
                pu=lenX*(.5f+.5f*radius*cosf(phi));
                pv=lenY*(.5f+.5f*radius*sinf(phi));
            }
            lsr->uv = make_float2(pu/lenX,pv/lenY);
            lsr->p = v+axisX*pu+axisY*pv;
            lsr->dir = lsr->p-shadingP;
            lsr->dist = length(lsr->dir);
            lsr->dir /= lsr->dist;
        }
        lsr->n=normal;
        lsr->NoL=dot(-lsr->dir,normal);
        // No retry or acceptance renormalization: a miss is a zero-weight draw
        // from the selected covering proposal. MIS evaluates that same density.
        lsr->PDF = lsr->dist > 0.0f && SpreadContains(p,lsr->uv,spread) ?
            ProposalPdf(p,lsr->dist) : 0.0f;
    }

    inline void SampleAsLight(LightSampleRecord* lsr, const float2& uu, const float3& shadingP) const {
        lsr->PDF = 0.0f;
        if (!(lenX > 0.0f && lenY > 0.0f)) return;
        SphericalEllipse ellipse;
        const bool spherical = ellipse.init(center() - shadingP, axisX, 0.5f * lenX, axisY, 0.5f * lenY);
        if (spherical) {
            lsr->dir = ellipse.sample(uu);
            const float denom = dot(lsr->dir, normal);
            if (denom == 0.0f) return;
            lsr->dist = dot(center() - shadingP, normal) / denom;
            if (!(lsr->dist > 0.0f) || !isfinite(lsr->dist)) return;
            lsr->p = shadingP + lsr->dist * lsr->dir;
            const float3 delta = lsr->p - v;
            lsr->uv = make_float2(dot(delta, axisX) / lenX, dot(delta, axisY) / lenY);
        } else {
            // Uniform ellipse-area fallback; every sample stays on the emitter.
            const float r = sqrtf(uu.x), phi = 2.0f * M_PIf * uu.y;
            lsr->uv = make_float2(0.5f + 0.5f * r * cosf(phi), 0.5f + 0.5f * r * sinf(phi));
            lsr->p = v + axisX * (lenX * lsr->uv.x) + axisY * (lenY * lsr->uv.y);
            lsr->dir = lsr->p - shadingP;
            lsr->dist = length(lsr->dir);
            if (!(lsr->dist > 0.0f)) return;
            lsr->dir /= lsr->dist;
        }
        lsr->n = normal;
        lsr->NoL = dot(-lsr->dir, normal);
        if (fabsf(lsr->NoL) > 0.0f) {
            lsr->PDF = spherical ? 1.0f / ellipse.solidAngle :
                AreaPdf(shadingP, lsr->dist);
            if (!isfinite(lsr->PDF)) lsr->PDF = 0.0f;
        }
    }

};
