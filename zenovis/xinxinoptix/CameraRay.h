#pragma once
#include "TraceStuff.h"

// Camera sampling shared by the path tracer and SPPM camera raygen.
static __forceinline__ __device__ void zenoCameraRay(uint3 idx, float2 subpixel_jitter,
    uint32_t& seed, float3& ray_origin, float3& ray_direction) {
    const auto& cam = params.cam;
    const auto w = params.width, h = params.height;
    const float aperture = clamp(cam.aperture, 0.0f, 100.0f);
    const float physical_aperture = aperture < 0.05f || aperture > 24.0f ? 0.0f : cam.focal_length / aperture;
    float2 d = 2.0f * make_float2(
        ( static_cast<float>( idx.x ) + subpixel_jitter.x ) / static_cast<float>( w ),
        ( static_cast<float>( idx.y ) + subpixel_jitter.y ) / static_cast<float>( h )
        ) - 1.0f;

    float2 r01 = {rnd(seed), rnd(seed)};

    float r0 = r01.x * 2.0f * M_PIf;
    float r1 = sqrtf(r01.y) * physical_aperture;

    float sin_yaw = sinf(cam.yaw);
    float cos_yaw = cosf(cam.yaw);
    float sin_pitch = sinf(cam.pitch);
    float cos_pitch = cosf(cam.pitch);

    mat3 tile_transform = mat3(
        cos_yaw, -sin_yaw * cos_pitch,  sin_pitch*sin_yaw,
        sin_yaw, cos_yaw * cos_pitch,   - cos_yaw * sin_pitch,
        0.0f,    sin_pitch,             cos_pitch
    );

    mat3 camera_transform = mat3(
        cam.right.x, cam.up.x, cam.front.x,
        cam.right.y, cam.up.y, cam.front.y,
        cam.right.z, cam.up.z, cam.front.z
    );

    // Under camer local space, cam.eye as origin, cam.right as X axis, cam.up as Y axis, cam.front as Z axis.
    float3 eye_shake     = r1 * (cosf(r0) * make_float3(1.0f,0.0f,0.0f) + sinf(r0) * make_float3(0.0f,1.0f,0.0f)); // r1 * ( cos(r0) , sin(r0) , 0 );
    float3 focal_plane_center = make_float3(cam.horizontal_shift*cam.width, cam.vertical_shift*cam.height, cam.focal_length);
    float3 old_direction =   focal_plane_center + make_float3(cam.width * 0.5f * d.x, cam.height * 0.5f * d.y, 0.0f);
    float3 tile_normal =  make_float3(sin_pitch*sin_yaw, - cos_yaw * sin_pitch, cos_pitch);

    float D = - dot(tile_normal , focal_plane_center);//surcface equaltion is Ax+By+Cz+D = 0

    /*to sphere coordinate
    x = r * sin(theta) * cos(phi) = r * C1;
    y = r * sin(theta) * sin(phi) = r * C2;
    z = r * cos(phi) = r* C3;
    */
    float old_r = length(old_direction);
    float3 C_vector = old_direction/old_r;
    float new_r = -D / dot(tile_normal,C_vector);
    /*
    Ax+By+Cz+D = A*C1*r + B*C2*r + C*C3*r + D = ((A,B,C) dot (C1,C2,C3)) * r + D =0
    old_direction/old_r = (C1,C2,C3)
    */
    float3 terminal_point = new_r * C_vector;
    terminal_point = terminal_point * (cam.focal_distance/cam.focal_length);//focal_length control

    //transform to world space
    terminal_point = camera_transform * terminal_point;
    eye_shake = camera_transform * eye_shake;

    ray_origin = eye_shake;
    ray_direction = terminal_point - eye_shake;
    ray_direction = normalize(ray_direction);
    if (params.physical_camera_panorama_camera) {
        ray_origin    = make_float3(0.0f, 0.0f, 0.0f);
        float phi = (float(idx.x) + subpixel_jitter.x) / float(w) * 2.0f * M_PIf;
        mat3 camera_transform = mat3(
                cam.right.x, cam.up.x, -cam.front.x,
                cam.right.y, cam.up.y, -cam.front.y,
                cam.right.z, cam.up.z, -cam.front.z
        );
        if (params.physical_camera_panorama_vr180) {
            int idxx = idx.x >= w/2? idx.x - w/2 : idx.x;
            phi = ((float(idxx) + subpixel_jitter.x) / float(w / 2) + 0.5f) * M_PIf;
            if (idx.x < w / 2) {
                ray_origin = camera_transform * make_float3(-params.physical_camera_pupillary_distance / 2.0f, 0.0f, 0.0f);
            }
            else {
                ray_origin = camera_transform * make_float3(params.physical_camera_pupillary_distance / 2.0f, 0.0f, 0.0f);
            }
        }
        float theta = (float(idx.y) + subpixel_jitter.y) / float(h) * M_PIf;
        float y = -cosf(theta);
        float z = sinf(theta) * cosf(phi);
        float x = sinf(theta) * sinf(-phi);

        ray_direction = camera_transform * make_float3(x, y, z);
    }

}
