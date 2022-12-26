#include<zeno.h>
using namespace zeno;

/**
 * @brief 给定两个向量，计算他们的法向（实际上就是叉乘向量再单位化）。
 * 
 * @param vec1 
 * @param vec2 
 * @return vec3f 
 */
inline vec3f calcNormal(const vec3f & vec1, const vec3f & vec2)
{
    auto res = cross(vec1, vec2);
    res = res / length(res);
    return res;
}

/**
 * @brief 归一化
 * 
 * @param vec 
 * @return vec3f 
 */
inline vec3f normalize(vec3f & vec)
{
    return vec/length(vec);
}