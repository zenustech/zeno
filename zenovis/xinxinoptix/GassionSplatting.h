#pragma once
#include "zxxglslvec.h"

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
    float* GetShBufferFormUniform(float4 *buffer,size_t index){
        return (float*) (buffer + index * 14);
    }

    static __inline__ __device__
    float GetOpacityFromUniform(float4 *buffer,size_t index){
        return buffer[index * 14+12].x;
    }

    __inline__ __device__
    float EvalGSOpacity(float4 *buffer,size_t index, vec3 dir, vec3 pos, const float *mat){
        //vec4 v0(mat[0],mat[4],mat[8],0);
        //vec4 v1(mat[1],mat[5],mat[9],0);
        //vec4 v2(mat[2],mat[6],mat[10],0);
        //vec4 v3(mat[3],mat[7],mat[11],1);
        vec4 v0(mat[0],mat[1],mat[2],mat[3]);
        vec4 v1(mat[4],mat[5],mat[6],mat[7]);
        vec4 v2(mat[8],mat[9],mat[10],mat[11]);

        vec4 new_pos = vec4(pos[0],pos[1],pos[2],1.0f);
        vec4 origin = vec4(0.0f,0.0f,0.0f,1.0f);
        new_pos = (*(mat4 *)mat)  * new_pos;
        vec3 new_origin;

        origin =  (*(mat4 *)mat) * origin;

        pos.x = new_pos.x;
        pos.y = new_pos.y;
        pos.z = new_pos.z;

        new_origin.x = origin.x;
        new_origin.y = origin.y;
        new_origin.z = origin.z;

        dir = pos - new_origin;

        dir = normalize(dir);

        pos = normalize(pos);


        //float op = GetOpacityFromUniform(buffer, index);
        float cosAlpha= dot(pos,dir);
        float cos2 = cosAlpha * cosAlpha;
        float sin2 = 1.0f - cos2;
        float result = 0.0f;

        result = exp(-8.0f * sin2) ;
        return result;

    }
        
        
        
    
    static __inline__ __device__
    vec3 EvalSH(float4* buffer, size_t index, int level, vec3 dir,const float *mat){
        vec3 color(1,0,0);
        dir = -vec3(mat[3],mat[7],mat[11]);
        dir = normalize(dir);
        float *SH_params =(float*) (buffer + index * 14);
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
        color = clamp(color,vec3(0,0,0),vec3(1,1,1));
        return color;
    }

}