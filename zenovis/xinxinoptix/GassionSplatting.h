#pragma once
#include "Host.h"
#include "zxxglslvec.h"
//#include "TraceStuff.h"

#define SH_C0 0.28209479177387814f

#define SH_C1 0.4886025119029199f

#define SH_C2_0 1.0925484305920792f
#define SH_C2_1 -1.0925484305920792f
#define SH_C2_2 0.31539156525252005f
#define SH_C2_3 -1.0925484305920792f
#define SH_C2_4 0.5462742152960396f

#define SH_C3_0 -0.5900435899266435f
#define SH_C3_1 2.890611442640554f
#define SH_C3_2 -0.4570457994644658f
#define SH_C3_3 0.3731763325901154f
#define SH_C3_4 -0.4570457994644658f
#define SH_C3_5 1.445305721320277f
#define SH_C3_6 -0.5900435899266435f

namespace GS{
    static __inline__ __device__
    vec3 GetParamFromBuffer(float *SH_params, int index){
        return vec3(SH_params[index*3],SH_params[index*3 +1],SH_params[index*3+2]);
    }

    static __inline__ __device__
    float GassionDensitiy(vec3 reletive_pos, mat3 conic_matrix){
        return 0.0f;
    }
    
    static __inline__ __device__
    vec3 EvalSH(float4* buffer, size_t index, int level, vec3 dir){
        vec3 color;
        dir = normalize(dir);
        float *SH_params =(float*) (buffer + index * 12);
        color = SH_C0 * GetParamFromBuffer(SH_params, 0);
        float x,y,z,xx,yy,zz,xy,xz,yz;
        x = dir.x;
        y = dir.y;
        z = dir.z;
        xx = x * x;
        yy = y * y;
        zz = z * z;
        xy = x * y;
        xz = x * z;
        yz = y * z;
        if(level > 0){
            color = color 
            - SH_C1 * GetParamFromBuffer(SH_params, 1) * y 
            + SH_C1 * GetParamFromBuffer(SH_params,2) * z
            - SH_C1 * GetParamFromBuffer(SH_params,3) * x;
            if(level >1) {
                color = color 
                + SH_C2_0 * xy * GetParamFromBuffer(SH_params,4) 
                + SH_C2_1 * yz * GetParamFromBuffer(SH_params,5) 
                + SH_C2_2 * (2.0f * zz - xx - yy) * GetParamFromBuffer(SH_params,6) 
                + SH_C2_3 * xz * GetParamFromBuffer(SH_params,7) 
                + SH_C2_4 * (xx - yy) * GetParamFromBuffer(SH_params,8);
                if(level > 2){
                    color = color
                    + SH_C3_0 * y * (3.0f * xx - yy) * GetParamFromBuffer(SH_params,9)
                    + SH_C3_1 * xy * z * GetParamFromBuffer(SH_params,10)
                    + SH_C3_2 * y * (4.0f * zz - xx - yy) * GetParamFromBuffer(SH_params,11)
                    + SH_C3_3 * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * GetParamFromBuffer(SH_params,12)
                    + SH_C3_4 * x * (4.0f * zz - xx - yy) * GetParamFromBuffer(SH_params,13)
                    + SH_C3_5 * z * (xx - yy) * GetParamFromBuffer(SH_params,14)
                    + SH_C3_6 * x * (xx - 3.0f * yy) * GetParamFromBuffer(SH_params,15);
                }
            }

        }
        color = color+0.5f;
        return color;
    }

}