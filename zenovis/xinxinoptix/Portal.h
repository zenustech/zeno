#pragma once

#ifndef __CUDACC_RTC__
#include <vector>
#include <algorithm>
#include <cassert>
#include <stdexcept>
#include <raiicuda.h>
#include <zeno/vec.h>

#include <zeno/utils/pfm.h>
#include <glm/gtc/matrix_transform.hpp>

#endif

#include <zeno/types/LightObject.h>
#include <Sampling.h>

struct Bounds2f {
    Vector2f pMin = Vector2f {FLT_MAX, FLT_MAX};
    Vector2f pMax = -Vector2f {FLT_MAX, FLT_MAX};

    bool contains(Vector2f p) {
        if (p[0] < pMin[0] || p[1] < pMin[1])
            return false;
        if (p[0] > pMax[0] || p[1] > pMax[1])
            return false;

        return true;
    }

    float area() {
        auto delta = pMax - pMin;
        return delta[0] * delta[1];
    }
};

namespace xx {

template <typename T>
struct Array2D {
    
#ifndef __CUDACC_RTC__
    std::vector<T> data;
    xinxinoptix::raii<CUdeviceptr> buffer;
#else
    T* data;
#endif
    uint32_t _x, _y;

    uint32_t XSize() const { return _x; }
    uint32_t YSize() const { return _y; }

    Array2D() = default;

#ifndef __CUDACC_RTC__

    Array2D(uint32_t x, uint32_t y) {
        _x = x; _y = y;
        data.resize(_x * _y, {});
    }

    auto upload() {

        size_t byte_size = sizeof(T) * data.size();

        buffer.resize( byte_size );
        cudaMemcpy((void*)buffer.handle, data.data(), byte_size, cudaMemcpyHostToDevice);

        struct Dummy {
            void* ptr;
            uint32_t _x, _y; 
        };

        return Dummy {
            (void*)buffer.handle, _x, _y
        };
    }

#endif

    T &operator()(uint32_t x, uint32_t y) { 
        size_t idx = x + y * _x;
        return data[idx];
    }

    const T &operator()(uint32_t x, uint32_t y) const { 
        size_t idx = x + y * _x;
        return data[idx];
    }

    T lookUV(float2 uv) const {

        auto xf = uv.x * _x;
        auto yf = uv.y * _y;
        xf -= 0.5f; yf -= 0.5f;

        auto xi = (int)floor(xf);
        auto yi = (int)floor(yf); 

        auto dx = xf - xi;
        auto dy = yf - yi;

        auto v00 = lookUp(  xi,   yi);
        auto v10 = lookUp(1+xi,   yi); 
        auto v01 = lookUp(  xi, 1+yi);
        auto v11 = lookUp(1+xi, 1+yi);

        return 
        
        (v00 * (1-dx) + v10 * dx) * (1-dy) 
             +
        (v01 * (1-dx) + v11 * dx) * dy;
    }

    T lookUp(int x, int y) const {

        if (x<0 || x >= _x) return {};
        if (y<0 || y >= _y) return {};
        size_t idx = x + y * _x;
        return data[idx];
    }
};

};

struct SummedAreaTable {
  public:
    // SummedAreaTable Public Methods
    SummedAreaTable() = default;

#ifndef __CUDACC_RTC__
    SummedAreaTable(const xx::Array2D<float> &values)
        : sum(values.XSize(), values.YSize()) {

        sum(0, 0) = values(0, 0);
        // Compute sums along first row and column
        for (int x = 1; x < sum.XSize(); ++x)
            sum(x, 0) = values(x, 0) + sum(x - 1, 0);
        for (int y = 1; y < sum.YSize(); ++y)
            sum(0, y) = values(0, y) + sum(0, y - 1);

        // Compute sums for the remainder of the entries
        for (int y = 1; y < sum.YSize(); ++y)
            for (int x = 1; x < sum.XSize(); ++x)
                sum(x, y) = (values(x, y) + sum(x - 1, y) + sum(x, y - 1) - sum(x - 1, y - 1));
    }

    auto upload() {
        return sum.upload();
    }

#endif

    float Integral(Bounds2f extent) const {
        double s =  ((double)Lookup(extent.pMax[0], extent.pMax[1]) - (double)Lookup(extent.pMin[0], extent.pMax[1])) 
                        +
                    ((double)Lookup(extent.pMin[0], extent.pMin[1]) - (double)Lookup(extent.pMax[0], extent.pMin[1]));
        return fmaxf(s / (sum.XSize() * sum.YSize()), 0);
    }

  private:
    // SummedAreaTable Private Methods
    float Lookup(float x, float y) const {
        // Rescale $(x,y)$ to table resolution and compute integer coordinates
        x = x * sum.XSize();
        y = y * sum.YSize();

        x = x - 0.5f;
        y = y - 0.5f;

        int x0 = (int)x; 
        int y0 = (int)y;

        float dx = x - int(x);
        float dy = y - int(y);

        // Bilinearly interpolate between surrounding table values
        float v00 = LookupInt(x0, y0), v10 = LookupInt(x0 + 1, y0);
        float v01 = LookupInt(x0, y0 + 1), v11 = LookupInt(x0 + 1, y0 + 1);
        
        return (1 - dx) * ( (1 - dy) * v00 + dy * v01 ) 
                    + 
                    dx  * ( (1 - dy) * v10 + dy * v11 );
    }

    float LookupInt(int x, int y) const {
        // Return zero at lower boundaries
        if (x <= 0 || y <= 0)
            return 0;

        // Reindex $(x,y)$ and return actual stored value
        x = min(x-1, (int)sum.XSize() - 1);
        y = min(y-1, (int)sum.YSize() - 1);
        return sum(x, y);
    }

    // SummedAreaTable Private Members
    xx::Array2D<double> sum;
};

template<typename T>
inline T BiLinear(T* data, uint width, uint height, float2 pos) {

    pos -= {0.5f, 0.5f};

    auto lowX = (int)std::floor(pos.x), highX = lowX+1;
    auto lowY = (int)std::floor(pos.y), highY = lowY+1;

    auto ratioX = pos.x - lowX;
    auto ratioY = pos.y - lowY;

    auto lookUp = [&](int x, int y) {
        if (x < 0 || x >= width ) return T{};
        if (y < 0 || y >= height) return T{};

        return *(data + (y * width + x));
    };

    auto v00 = lookUp(lowX, lowY);
    auto v10 = lookUp(highX,lowY);
    auto vv0 = v00 * (1-ratioX) + v10 * ratioX;

    auto v01 = lookUp(lowX, highY);
    auto v11 = lookUp(highX,highY);
    auto vv1 = v01 * (1-ratioX) + v11 * ratioX;

    return vv0 * (1-ratioY) + vv1 * ratioY;
}

struct Portal {
    Vector3f p0, p1, p2, p3;
    uint32_t psize;
};

struct PortalLight {

    xx::Array2D<float3> image {};
    xx::Array2D<float>  dist  {};
    SummedAreaTable sat;

    Portal portal;
    Vector3f X,Y,Z;
    //PortalLight() = default;

    auto luminance(float3 c) {
        return dot(c, float3{0.2722287, 0.6740818, 0.0536895});
    };

#ifndef __CUDACC_RTC__

    auto pack() {

        auto image_dummy = image.upload();
        auto dist_dummy = dist.upload();
        auto sat_dummy = sat.upload();

        struct Dummy {
            
            decltype (image_dummy) _image; 
            decltype (dist_dummy) _dist;
            decltype (sat_dummy) _sat;
            Portal portal;
            Vector3f X,Y,Z;
        };

        return Dummy {  image_dummy, dist_dummy, sat_dummy, portal, X, Y, Z };
    }

    PortalLight(const Portal& por, float3* texture, uint tex_width, uint tex_height, glm::mat4* rotate=nullptr) : portal(por) {

        Vector3f p01 = normalize(portal.p1 - portal.p0);
        Vector3f p12 = normalize(portal.p2 - portal.p1);
        Vector3f p32 = normalize(portal.p2 - portal.p3);
        Vector3f p03 = normalize(portal.p3 - portal.p0);
        // Do opposite edges have the same direction?
        if (std::abs(dot(p01, p32) - 1) > .001 || std::abs(dot(p12, p03) - 1) > .001)
            throw std::runtime_error("Infinite light portal isn't a planar quadrilateral");

        // Sides perpendicular?
        if (std::abs(dot(p01, p12)) > .001 || std::abs(dot(p12, p32)) > .001 ||
            std::abs(dot(p32, p03)) > .001 || std::abs(dot(p03, p01)) > .001)
            throw std::runtime_error("Infinite light portal isn't a planar quadrilateral");

        X = p03, Y = p01, Z = -cross(X, Y);

        uint pixel_count_x = por.psize, pixel_count_y = por.psize;
        uint pixel_count = pixel_count_x * pixel_count_y;

        image = xx::Array2D<float3>(pixel_count_x, pixel_count_y);
        dist = xx::Array2D<float>(pixel_count_x, pixel_count_y);

        for (uint i=0; i<pixel_count_x; ++i) {
            for (uint j=0; j<pixel_count_y; ++j) {

                Vector2f uv { (i+0.5f)/pixel_count_x, (j+0.5f)/pixel_count_y };
                float duv_dw;
                zeno::vec3f world_dir = uv_direction(reinterpret_cast<float2&>(uv), &duv_dw);

                if (rotate != nullptr && *rotate != glm::mat4(1.0f)) {
                    glm::vec4 tmp = glm::vec4(world_dir[0], world_dir[1], world_dir[2], 0.0f);
                    tmp = tmp * (*rotate);

                    world_dir = {tmp.x, tmp.y, tmp.z};
                }

                auto suv = sphereUV(reinterpret_cast<float3&>(world_dir), true);
                auto pos = (*(float2*)&suv) * make_float2(tex_width, tex_height);

                auto pixel = BiLinear(texture, tex_width, tex_height, pos);
                auto average = this->luminance(pixel);
                //average *= std::sin(M_PIf * suv.y);

                image(i, j) = pixel;
                dist(i, j) = duv_dw * average;
            } // j
        } // i

        sat = SummedAreaTable(dist);

    #if !defined( NDEBUG ) 
        zeno::write_pfm("portal.pfm", image.XSize(), image.YSize(), (float*)image.data.data());
        zeno::write_pfm("dist.pfm", dist.XSize(), dist.YSize(), (float*)dist.data.data(), true);
    #endif
    }

#endif

    inline float area() {
        auto a = length(portal.p1 - portal.p0);
        auto b = length(portal.p2 - portal.p1);
        return a * b;
    }

    float phi() {

        float3 sum {};
        for (uint y=0; y<image.YSize(); ++y) {
            for (uint x=0; x<image.XSize(); ++x) {
                auto value = image.lookUp(x, y);

                float2 uv = { (x+0.5f)/image.XSize(), 
                              (y+0.5f)/image.YSize()};
                float duvdw;
                uv_direction(uv, &duvdw);
                sum += value / duvdw;
            }
        }

        sum /= (image.XSize() * image.YSize());
        return area() * this->luminance(sum);
    }

    float3 Le(const Vector3f& ray_origin, const Vector3f& ray_dir) {
        Bounds2f bds;
        auto valid = ImageBounds(ray_origin, bds);
        if (!valid) {return {};}

        auto x = dot(ray_dir, X);
        auto y = dot(ray_dir, Y);
        auto z = dot(ray_dir, Z);

        if (z <= 0) {return {};}

        auto sinL = sqrt(1.0f - z * z);
        auto angleX = asin(x / sinL);
        auto angleY = -acos(y / sinL);

         Vector2f uv = {
            ( angleX + M_PI_2f ) / M_PIf,
            ( angleY + M_PIf ) / M_PIf
        };

        if (bds.contains(uv)) {
            return image.lookUp(uv[0], uv[1]);
        } else {
            return {};
        }
    }

    bool ImageBounds(const Vector3f& p, Bounds2f& bounds) const {

        auto v0 = normalize(portal.p0 - p);
        auto v1 = normalize(portal.p2 - p);

        auto x0 = dot(v0, X);
        auto y0 = dot(v0, Y);
        auto z0 = dot(v0, Z);

        if (z0 <= 0) { return false; }

        auto angleX0 = atan2(x0, z0);
        auto angleY0 = atan2(y0, z0);

        auto x1 = dot(v1, X);
        auto y1 = dot(v1, Y);
        auto z1 = dot(v1, Z);
        
        if (z1 <= 0) { return false; }

        auto angleX1 = atan2(x1, z1);
        auto angleY1 = atan2(y1, z1);

        if (angleX0 >= angleX1 || angleY0 >= angleY1) 
        {
            return false;
        } 

        Vector2f uv0 = { 
            ( angleX0 + M_PI_2f ) / M_PIf,
            ( angleY0 + M_PI_2f ) / M_PIf
        };

        Vector2f uv1 = {
            ( angleX1 + M_PI_2f ) / M_PIf,
            ( angleY1 + M_PI_2f ) / M_PIf
        };

        bounds = Bounds2f{ uv0, uv1 };
        return true; 
    }

    template <typename CDF>
    static float SampleBisection(CDF P, const float u, float min, float max, uint n) {
        assert(0.0<=min && min < max && max<=1.0);

        while (min < max && ( (n * max) - (n * min)) > 1) {

            DCHECK(P(min) <= u);
            DCHECK(P(max) >= u);
            float mid = (min + max) / 2;
            auto PM = P(mid);
            //PM = clamp(PM, 0.0f, 1.0f); 

            if (PM > u)
                max = mid;
            else
                min = mid;
        }

        // Find sample by interpolating between _min_ and _max_
        float t = (u - P(min)) / (P(max) - P(min));
        return clamp(pbrt::Lerp(t, min, max), min, max);
    }

    float Eval(float2 p) const {
        float2 pi{ fminf(p.x * dist.XSize(), dist.XSize() - 1),
                   fminf(p.y * dist.YSize(), dist.YSize() - 1) };
        //return dist.lookUp((int)pi.x, (int)pi.y);
        return dist.lookUV(p);
    }

    Vector2f direction_uv(Vector3f dir, float *duvdw=nullptr) {

        auto x = dot(dir, X);
        auto y = dot(dir, Y);
        auto z = dot(dir, Z);

        if (z <= 0) {return {};}

        auto w = Vector3f{x, y, z};

        if (duvdw)
            *duvdw = pbrt::Sqr(M_PIf) * (1 - pbrt::Sqr(w[0])) * (1 - pbrt::Sqr(w[1])) / w[2]; 

        auto sinL = sqrt(1.0f - z * z);
        auto angleX = asin(x / sinL);
        auto angleY = -acos(y / sinL);

         Vector2f uv = {
            ( angleX + M_PI_2f ) / M_PIf,
            ( angleY + M_PIf ) / M_PIf
        };
    }

    Vector3f uv_direction(float2 uv, float* duvdw=nullptr) {

        float alpha = -M_PIf / 2 + uv.x * M_PIf; 
        float beta  = -M_PIf / 2 + uv.y * M_PIf;
        float x = tanf(alpha), y = tanf(beta);

        DCHECK(!IsInf(x) && !IsInf(y));

        Vector3f w = normalize(Vector3f(x, y, 1));
        
        if (duvdw)
            *duvdw = pbrt::Sqr(M_PIf) * (1 - pbrt::Sqr(w[0])) * (1 - pbrt::Sqr(w[1])) / w[2];
        
        Vector3f dir {};
        dir = dir + X * w[0];
        dir = dir + Y * w[1];
        dir = dir + Z * w[2];
        return dir;
    }

    void sample(LightSampleRecord& lsr, const Vector3f& pos, float2 uu, float3& color) {
        Bounds2f bds; // uv bounds
        auto valid = ImageBounds(pos, bds);
        if (!valid) return;

        auto bIntegral = sat.Integral(bds);
        if( bIntegral <= 0 ) return;

        auto Px = [&](float x) -> float {
            Bounds2f bx = bds;
            bx.pMax[0] = x;
            return sat.Integral(bx) / bIntegral;
        };

        float2 uv;
        uv.x = SampleBisection(Px, uu.x, bds.pMin[0], bds.pMax[0], image.XSize());

        uint nx = image.XSize();
        Bounds2f bCond { 
            { floor(uv.x * nx)/nx, bds.pMin[1] },
            { ceil (uv.x * nx)/nx, bds.pMax[1] } };

        if (bCond.pMin[0] == bCond.pMax[0])
            bCond.pMax[0] += 1.0f / nx;

        float condIntegral = sat.Integral(bCond);
        if (condIntegral == 0) 
            return;

        auto Py = [&](float y) -> float {
            Bounds2f by = bCond;
            by.pMax[1] = y;
            return sat.Integral(by) / condIntegral;
        };
        uv.y = SampleBisection(Py, uu.y, bds.pMin[1], bds.pMax[1], image.YSize());
        //uv = clamp(uv, 0.0, 1.0);

        float duvdw;
        auto tmp = uv_direction(uv, &duvdw);

        lsr.dir = reinterpret_cast<float3&>(tmp);
        lsr.dist = FLT_MAX;
        lsr.uv = uv;

        // Compute PDF and return point sampled from windowed function
        lsr.PDF = Eval(uv) / bIntegral;
        lsr.PDF /= duvdw;
        if(!isfinite(lsr.PDF)) {
            lsr.PDF = 0.0;
            return;
        }

        color = image.lookUV(uv);
    }

    float PDF(Vector3f p, Vector3f w) {
        float duvdw;
        auto uv = direction_uv(w, &duvdw);

        Bounds2f bds;
        bool valid = ImageBounds(p, bds);
        if (!valid) return 0.0f;

        float integ = sat.Integral(bds);
        if (integ == 0) return 0.0f;

        return Eval(reinterpret_cast<float2&>(uv)) / duvdw;
    }
};

struct PortalLightList {

#ifndef __CUDACC_RTC__
    std::vector<PortalLight> list;
    xinxinoptix::raii<CUdeviceptr> buffer;

    std::vector<float> pdf;
    std::vector<float> cdf;
    xinxinoptix::raii<CUdeviceptr> pdf_buffer;
    xinxinoptix::raii<CUdeviceptr> cdf_buffer;

    xinxinoptix::raii<CUdeviceptr> dummy_buffer;
#else
    PortalLight *list;
    size_t count;

    float* pdf;
    float* cdf;
#endif

    inline size_t COUNT() {
    #ifndef __CUDACC_RTC__
        return list.size();
    #else
        return count;
    #endif
    }

#ifndef __CUDACC_RTC__
    auto upload() {

        if (list.size() == 0) {
            *this = {};
            return 0llu;
        }

        auto first = list.front().pack();
        std::vector<decltype(first)> tmp;
        tmp.reserve(list.size());
        tmp.push_back(first);

        pdf.clear(); pdf.resize(list.size());
        cdf.clear(); cdf.resize(list.size());

        auto power = list.front().phi();
        pdf[0] = power;
        cdf[0] = power;

        for (size_t i=1; i<list.size(); ++i) {
            auto ele = list[i].pack();
            tmp.push_back(ele);

            auto phi = list[i].phi();

            power += phi;
            pdf[i] = phi;
            cdf[i] = power;
        }

        for (size_t i=0; i<list.size(); ++i) {
            pdf[i] /= power;
            cdf[i] /= power;
        }
        
        size_t byte_size = sizeof(first) * list.size();

        buffer.resize(byte_size);
        cudaMemcpy((void*)buffer.handle, tmp.data(), byte_size, cudaMemcpyHostToDevice);

        byte_size = sizeof(float) * list.size();
        pdf_buffer.resize(byte_size);
        cudaMemcpy((void*)pdf_buffer.handle, pdf.data(), byte_size, cudaMemcpyHostToDevice);

        cdf_buffer.resize(byte_size);
        cudaMemcpy((void*)cdf_buffer.handle, cdf.data(), byte_size, cudaMemcpyHostToDevice);

        struct Dummy {
            void* data; 
            size_t count;
            void* pdf; void* cdf;
        };

        auto result =  Dummy { (void*)buffer.handle, list.size(), 
                                       (void*)pdf_buffer.handle, (void*)cdf_buffer.handle };

        dummy_buffer.resize(sizeof(result));
        cudaMemcpy((void*)dummy_buffer.handle, &result, sizeof(result), cudaMemcpyHostToDevice);

        return dummy_buffer.handle;
    }
#endif

};

struct DistantLightList {

#ifndef __CUDACC_RTC__
    std::vector<zeno::DistantLightData> list;
    std::vector<float> cdf;

    xinxinoptix::raii<CUdeviceptr> data_buffer;
    xinxinoptix::raii<CUdeviceptr> cdf_buffer;

    xinxinoptix::raii<CUdeviceptr> dummy_buffer;
#else
    zeno::DistantLightData* list;
    float* cdf;
    uint count;
#endif
    inline size_t COUNT() {
#ifndef __CUDACC_RTC__
        return list.size();
#else   
        return count;
#endif
    }

#ifndef __CUDACC_RTC__

    auto upload() {

        size_t byte_size  = sizeof(zeno::DistantLightData) * list.size();
        data_buffer.resize( byte_size );
        cudaMemcpy((void*)data_buffer.handle, list.data(), byte_size, cudaMemcpyHostToDevice);

        byte_size = sizeof(float) * cdf.size();
        cdf_buffer.resize(byte_size);
        cudaMemcpy((void*)cdf_buffer.handle, cdf.data(), byte_size, cudaMemcpyHostToDevice);

        struct Dummy {
            void* data;
            void* cdf;
            size_t count;
        };

        Dummy dummy {
            (void*)data_buffer.handle,
            (void*)cdf_buffer.handle,
            list.size()
        };

        dummy_buffer.resize(sizeof(dummy));
        cudaMemcpy((void*)dummy_buffer.handle, &dummy, sizeof(dummy), cudaMemcpyHostToDevice);

        return dummy_buffer.handle;     
    }

#endif
};