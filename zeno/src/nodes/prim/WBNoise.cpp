//
// Created by WangBo on 2022/7/5.
//

#include <zeno/zeno.h>
#include <zeno/types/PrimitiveObject.h>
#include <zeno/utils/log.h>
#include <glm/gtx/quaternion.hpp>
#include <cmath>
#include <random>
//#include <array>

namespace zeno
{
namespace
{

///////////////////////////////////////////////////////////////////////////////
// 2022.07.11 Perlin Noise
///////////////////////////////////////////////////////////////////////////////

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Perlin Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
const int noise_permutation[] = {
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
    151,160,137,91,90,15,
    131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
    190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
    88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
    77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
    102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
    135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
    5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
    223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
    129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
    251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
    49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
    138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180,
};

float noise_fade(float t) {
    // Fade function as defined by Ken Perlin.  This eases coordinate values
    // so that they will ease towards integral values.  This ends up smoothing
    // the final output.
    return t * t * t * (t * (t * 6 - 15) + 10);         // 6t^5 - 15t^4 + 10t^3
}

int noise_inc(int num) {
    return num + 1;
}

float noise_grad(int hash, float x, float y, float z) {
    switch (hash & 0xF) {
    case 0x0: return  x + y;
    case 0x1: return -x + y;
    case 0x2: return  x - y;
    case 0x3: return -x - y;
    case 0x4: return  x + z;
    case 0x5: return -x + z;
    case 0x6: return  x - z;
    case 0x7: return -x - z;
    case 0x8: return  y + z;
    case 0x9: return -y + z;
    case 0xA: return  y - z;
    case 0xB: return -y - z;
    case 0xC: return  y + x;
    case 0xD: return -y + z;
    case 0xE: return  y - x;
    case 0xF: return -y - z;
    default: return 0;
    }
}

float noise_perlin(float x, float y, float z)
{
    x = fract(x / 256.f) * 256.f;
    y = fract(y / 256.f) * 256.f;
    z = fract(z / 256.f) * 256.f;

    int xi = (int)x & 255;          // Calculate the "unit cube" that the point asked will be located in
    int yi = (int)y & 255;          // The left bound is ( |_x_|,|_y_|,|_z_| ) and the right bound is that
    int zi = (int)z & 255;          // plus 1.  Next we calculate the location (from 0.0 to 1.0) in that cube.

    float xf = x - (int)x;
    float yf = y - (int)y;
    float zf = z - (int)z;

    float u = noise_fade(xf);
    float v = noise_fade(yf);
    float w = noise_fade(zf);

    int aaa = noise_permutation[noise_permutation[noise_permutation[xi] + yi] + zi];
    int aba = noise_permutation[noise_permutation[noise_permutation[xi] + noise_inc(yi)] + zi];
    int aab = noise_permutation[noise_permutation[noise_permutation[xi] + yi] + noise_inc(zi)];
    int abb = noise_permutation[noise_permutation[noise_permutation[xi] + noise_inc(yi)] + noise_inc(zi)];
    int baa = noise_permutation[noise_permutation[noise_permutation[noise_inc(xi)] + yi] + zi];
    int bba = noise_permutation[noise_permutation[noise_permutation[noise_inc(xi)] + noise_inc(yi)] + zi];
    int bab = noise_permutation[noise_permutation[noise_permutation[noise_inc(xi)] + yi] + noise_inc(zi)];
    int bbb = noise_permutation[noise_permutation[noise_permutation[noise_inc(xi)] + noise_inc(yi)] + noise_inc(zi)];

    float x1 = mix(noise_grad(aaa, xf, yf, zf),
        noise_grad(baa, xf - 1, yf, zf),
        u);
    float x2 = mix(noise_grad(aba, xf, yf - 1, zf),
        noise_grad(bba, xf - 1, yf - 1, zf),
        u);
    float y1 = mix(x1, x2, v);
    x1 = mix(noise_grad(aab, xf, yf, zf - 1),
        noise_grad(bab, xf - 1, yf, zf - 1),
        u);
    x2 = mix(noise_grad(abb, xf, yf - 1, zf - 1),
        noise_grad(bbb, xf - 1, yf - 1, zf - 1),
        u);
    float y2 = mix(x1, x2, v);

    return mix(y1, y2, w);
}

struct erode_noise_perlin : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3") terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float") terrain->add_attr<float>(attrName);
        }

        auto vec3fAttrName = get_input<StringObject>("vec3fAttrName")->get();
        if (!terrain->verts.has_attr(vec3fAttrName))
        {
            zeno::log_error("no such data named '{}'.", vec3fAttrName);
        }
        auto& vec3fAttr = terrain->verts.attr<vec3f>(vec3fAttrName);


        terrain->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++)
            {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>)
                {
                    float x = noise_perlin(vec3fAttr[i][0], vec3fAttr[i][1], vec3fAttr[i][2]);
                    float y = noise_perlin(vec3fAttr[i][1], vec3fAttr[i][2], vec3fAttr[i][0]);
                    float z = noise_perlin(vec3fAttr[i][2], vec3fAttr[i][0], vec3fAttr[i][1]);
                    arr[i] = vec3f(x, y, z);
                }
                else
                {
                    arr[i] = noise_perlin(vec3fAttr[i][0], vec3fAttr[i][1], vec3fAttr[i][2]);
                }
            }
            });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_noise_perlin,
    { /* inputs: */ {
            "prim_2DGrid",
            {"string", "vec3fAttrName", "pos"},
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
            {"string", "attrName", "noise"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Simplex Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
int noise_fastfloor(double x) {
    return x > 0 ? (int)x : (int)x - 1;
}

const int noise_simplex[][4] = {
    {0,1,2,3},{0,1,3,2},{0,0,0,0},{0,2,3,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,2,3,0},
    {0,2,1,3},{0,0,0,0},{0,3,1,2},{0,3,2,1},{0,0,0,0},{0,0,0,0},{0,0,0,0},{1,3,2,0},
    {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
    {1,2,0,3},{0,0,0,0},{1,3,0,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,3,0,1},{2,3,1,0},
    {1,0,2,3},{1,0,3,2},{0,0,0,0},{0,0,0,0},{0,0,0,0},{2,0,3,1},{0,0,0,0},{2,1,3,0},
    {0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},
    {2,0,1,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,0,1,2},{3,0,2,1},{0,0,0,0},{3,1,2,0},
    {2,1,0,3},{0,0,0,0},{0,0,0,0},{0,0,0,0},{3,1,0,2},{0,0,0,0},{3,2,0,1},{3,2,1,0}
};

float noise_sGrad3(int hash, float x, float y, float z) {
    switch (hash & 0xF) {
    case 0x0: return  x + y;
    case 0x1: return -x + y;
    case 0x2: return  x - y;
    case 0x3: return -x - y;
    case 0x4: return  x + z;
    case 0x5: return -x + z;
    case 0x6: return  x - z;
    case 0x7: return -x - z;
    case 0x8: return  y + z;
    case 0x9: return -y + z;
    case 0xA: return  y - z;
    case 0xB: return -y - z;
    case 0xC: return  y + x;
    case 0xD: return -y + z;
    case 0xE: return  y - x;
    case 0xF: return -y - z;
    default: return 0;
    }
}

float noise_sGrad4(int hash, float x, float y, float z, float w) {
    switch (hash & 0x1F) {
    case 0x00: return  y + z + w;
    case 0x01: return  y + z - w;
    case 0x02: return  y - z + w;
    case 0x03: return  y - z - w;
    case 0x04: return -y + z + w;
    case 0x05: return -y + z - w;
    case 0x06: return -y - z + w;
    case 0x07: return -y - z - w;

    case 0x08: return  x + z + w;
    case 0x09: return  x + z - w;
    case 0x0A: return  x - z + w;
    case 0x0B: return  x - z - w;
    case 0x0C: return -x + z + w;
    case 0x0D: return -x + z - w;
    case 0x0E: return -x - z + w;
    case 0x0F: return -x - z - w;

    case 0x10: return  x + y + w;
    case 0x11: return  x + y - w;
    case 0x12: return  x - y + w;
    case 0x13: return  x - y - w;
    case 0x14: return -x + y + w;
    case 0x15: return -x + y - w;
    case 0x16: return -x - y + w;
    case 0x17: return -x - y - w;

    case 0x18: return  x + y + z;
    case 0x19: return  x + y - z;
    case 0x1A: return  x - y + z;
    case 0x1B: return  x - y - z;
    case 0x1C: return -x + y + z;
    case 0x1D: return -x + y - z;
    case 0x1E: return -x - y + z;
    case 0x1F: return -x - y - z;
    default: return 0;
    }
}

// 3D Perlin simplex noise
// @param[in] x float coordinate
// @param[in] y float coordinate
// @param[in] z float coordinate
// @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
float noise_simplexNoise3(float x, float y, float z) {
    float n0, n1, n2, n3; // Noise contributions from the four corners

    // Skewing/Unskewing factors for 3D
    static const float F3 = 1.0f / 3.0f;
    static const float G3 = 1.0f / 6.0f;

    // Skew the input space to determine which simplex cell we're in
    float s = (x + y + z) * F3; // Very nice and simple skew factor for 3D
    int i = noise_fastfloor(x + double(s));
    int j = noise_fastfloor(y + double(s));
    int k = noise_fastfloor(z + double(s));
    float t = (float)(i + j + k) * G3;
    float X0 = (float)i - t; // Unskew the cell origin back to (x,y,z) space
    float Y0 = (float)j - t;
    float Z0 = (float)k - t;
    float x0 = x - X0; // The x,y,z distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;

    // For the 3D case, the simplex shape is a slightly irregular tetrahedron.
    // Determine which simplex we are in.
    int i1, j1, k1; // Offsets for second corner of simplex in (i,j,k) coords
    int i2, j2, k2; // Offsets for third corner of simplex in (i,j,k) coords
    if (x0 >= y0) {
        if (y0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 1; k2 = 0; // X Y Z order
        }
        else if (x0 >= z0) {
            i1 = 1; j1 = 0; k1 = 0; i2 = 1; j2 = 0; k2 = 1; // X Z Y order
        }
        else {
            i1 = 0; j1 = 0; k1 = 1; i2 = 1; j2 = 0; k2 = 1; // Z X Y order
        }
    }
    else { // x0<y0
        if (y0 < z0) {
            i1 = 0; j1 = 0; k1 = 1; i2 = 0; j2 = 1; k2 = 1; // Z Y X order
        }
        else if (x0 < z0) {
            i1 = 0; j1 = 1; k1 = 0; i2 = 0; j2 = 1; k2 = 1; // Y Z X order
        }
        else {
            i1 = 0; j1 = 1; k1 = 0; i2 = 1; j2 = 1; k2 = 0; // Y X Z order
        }
    }

    // A step of (1,0,0) in (i,j,k) means a step of (1-c,-c,-c) in (x,y,z),
    // a step of (0,1,0) in (i,j,k) means a step of (-c,1-c,-c) in (x,y,z), and
    // a step of (0,0,1) in (i,j,k) means a step of (-c,-c,1-c) in (x,y,z), where
    // c = 1/6.
    float x1 = x0 - (float)i1 + G3; // Offsets for second corner in (x,y,z) coords
    float y1 = y0 - (float)j1 + G3;
    float z1 = z0 - (float)k1 + G3;
    float x2 = x0 - (float)i2 + 2.0f * G3; // Offsets for third corner in (x,y,z) coords
    float y2 = y0 - (float)j2 + 2.0f * G3;
    float z2 = z0 - (float)k2 + 2.0f * G3;
    float x3 = x0 - 1.0f + 3.0f * G3; // Offsets for last corner in (x,y,z) coords
    float y3 = y0 - 1.0f + 3.0f * G3;
    float z3 = z0 - 1.0f + 3.0f * G3;

    // Wrap the integer indices at 256, to avoid indexing permutation[] out of bounds
    int ii = i & 0xff;
    int jj = j & 0xff;
    int kk = k & 0xff;

    // Work out the hashed gradient indices of the four simplex corners
    int gi0 = noise_permutation[ii + noise_permutation[jj + noise_permutation[kk]]];
    int gi1 = noise_permutation[ii + i1 + noise_permutation[jj + j1 + noise_permutation[kk + k1]]];
    int gi2 = noise_permutation[ii + i2 + noise_permutation[jj + j2 + noise_permutation[kk + k2]]];
    int gi3 = noise_permutation[ii + 1 + noise_permutation[jj + 1 + noise_permutation[kk + 1]]];

    // Calculate the contribution from the four corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0;
    //    float t0 = 0.5f - x0*x0 - y0*y0 - z0*z0;
    if (t0 < 0) {
        n0 = 0.0;
    }
    else {
        t0 *= t0;
        n0 = t0 * t0 * noise_sGrad3(gi0, x0, y0, z0);
    }
    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1;
    //    float t1 = 0.5f - x1*x1 - y1*y1 - z1*z1;
    if (t1 < 0) {
        n1 = 0.0;
    }
    else {
        t1 *= t1;
        n1 = t1 * t1 * noise_sGrad3(gi1, x1, y1, z1);
    }
    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2;
    //    float t2 = 0.5f - x2*x2 - y2*y2 - z2*z2;
    if (t2 < 0) {
        n2 = 0.0;
    }
    else {
        t2 *= t2;
        n2 = t2 * t2 * noise_sGrad3(gi2, x2, y2, z2);
    }
    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3;
    //    float t3 = 0.5f - x3*x3 - y3*y3 - z3*z3;
    if (t3 < 0) {
        n3 = 0.0;
    }
    else {
        t3 *= t3;
        n3 = t3 * t3 * noise_sGrad3(gi3, x3, y3, z3);
    }
    // Add contributions from each corner to get the final noise value.
    // The result is scaled to stay just inside [-1,1]
    return 32.0f * (n0 + n1 + n2 + n3);
}

// 4D Perlin simplex noise
// @param[in] x float coordinate
// @param[in] y float coordinate
// @param[in] z float coordinate
// @param[in] w float coordinate
// @return Noise value in the range[-1; 1], value of 0 on all integer coordinates.
float noise_simplexNoise4(float x, float y, float z, float w) {

    float n0, n1, n2, n3, n4;   // Noise contributions from the five corners

    static const float F4 = 0.309016994f;   // F4 = (sqrt(5) - 1) / 4
    static const float G4 = 0.138196601f;   // G4 = (5 - sqrt(5)) / 20

    // Skew the (x,y,z,w) space to determine which cell of 24 simplices we're in
    float s = (x + y + z + w) * F4; // Factor for 4D skewing
    float xs = x + s;
    float ys = y + s;
    float zs = z + s;
    float ws = w + s;
    int i = noise_fastfloor(xs);
    int j = noise_fastfloor(ys);
    int k = noise_fastfloor(zs);
    int l = noise_fastfloor(ws);

    float t = (float)(i + j + k + l) * G4; // Factor for 4D unskewing
    float X0 = (float)i - t; // Unskew the cell origin back to (x,y,z,w) space
    float Y0 = (float)j - t;
    float Z0 = (float)k - t;
    float W0 = (float)l - t;

    float x0 = x - X0;  // The x,y,z,w distances from the cell origin
    float y0 = y - Y0;
    float z0 = z - Z0;
    float w0 = w - W0;

    // For the 4D case, the simplex is a 4D shape I won't even try to describe.
    // To find out which of the 24 possible simplices we're in, we need to
    // determine the magnitude ordering of x0, y0, z0 and w0.
    // The method below is a good way of finding the ordering of x,y,z,w and
    // then find the correct traversal order for the simplex we're in.
    // First, six pair-wise comparisons are performed between each possible pair
    // of the four coordinates, and the results are used to add up binary bits
    // for an integer index.
    int c1 = (x0 > y0) ? 32 : 0;
    int c2 = (x0 > z0) ? 16 : 0;
    int c3 = (y0 > z0) ? 8 : 0;
    int c4 = (x0 > w0) ? 4 : 0;
    int c5 = (y0 > w0) ? 2 : 0;
    int c6 = (z0 > w0) ? 1 : 0;
    int c = c1 + c2 + c3 + c4 + c5 + c6;

    int i1, j1, k1, l1; // The integer offsets for the second simplex corner
    int i2, j2, k2, l2; // The integer offsets for the third simplex corner
    int i3, j3, k3, l3; // The integer offsets for the fourth simplex corner

    // simplex[c] is a 4-vector with the numbers 0, 1, 2 and 3 in some order.
    // Many values of c will never occur, since e.g. x>y>z>w makes x<z, y<w and x<w
    // impossible. Only the 24 indices which have non-zero entries make any sense.
    // We use a thresholding to set the coordinates in turn from the largest magnitude.
    // The number 3 in the "simplex" array is at the position of the largest coordinate.
    i1 = noise_simplex[c][0] >= 3 ? 1 : 0;
    j1 = noise_simplex[c][1] >= 3 ? 1 : 0;
    k1 = noise_simplex[c][2] >= 3 ? 1 : 0;
    l1 = noise_simplex[c][3] >= 3 ? 1 : 0;
    // The number 2 in the "simplex" array is at the second largest coordinate.
    i2 = noise_simplex[c][0] >= 2 ? 1 : 0;
    j2 = noise_simplex[c][1] >= 2 ? 1 : 0;
    k2 = noise_simplex[c][2] >= 2 ? 1 : 0;
    l2 = noise_simplex[c][3] >= 2 ? 1 : 0;
    // The number 1 in the "simplex" array is at the second smallest coordinate.
    i3 = noise_simplex[c][0] >= 1 ? 1 : 0;
    j3 = noise_simplex[c][1] >= 1 ? 1 : 0;
    k3 = noise_simplex[c][2] >= 1 ? 1 : 0;
    l3 = noise_simplex[c][3] >= 1 ? 1 : 0;
    // The fifth corner has all coordinate offsets = 1, so no need to look that up.

    float x1 = x0 - (float)i1 + G4; // Offsets for second corner in (x,y,z,w) coords
    float y1 = y0 - (float)j1 + G4;
    float z1 = z0 - (float)k1 + G4;
    float w1 = w0 - (float)l1 + G4;
    float x2 = x0 - (float)i2 + 2.0f * G4; // Offsets for third corner in (x,y,z,w) coords
    float y2 = y0 - (float)j2 + 2.0f * G4;
    float z2 = z0 - (float)k2 + 2.0f * G4;
    float w2 = w0 - (float)l2 + 2.0f * G4;
    float x3 = x0 - (float)i3 + 3.0f * G4; // Offsets for fourth corner in (x,y,z,w) coords
    float y3 = y0 - (float)j3 + 3.0f * G4;
    float z3 = z0 - (float)k3 + 3.0f * G4;
    float w3 = w0 - (float)l3 + 3.0f * G4;
    float x4 = x0 - 1.0f + 4.0f * G4; // Offsets for last corner in (x,y,z,w) coords
    float y4 = y0 - 1.0f + 4.0f * G4;
    float z4 = z0 - 1.0f + 4.0f * G4;
    float w4 = w0 - 1.0f + 4.0f * G4;

    // Wrap the integer indices at 256, to avoid indexing permutation[] out of bounds
    int ii = i & 0xff;
    int jj = j & 0xff;
    int kk = k & 0xff;
    int ll = l & 0xff;

    // Calculate the contribution from the five corners
    float t0 = 0.6f - x0 * x0 - y0 * y0 - z0 * z0 - w0 * w0;
    if (t0 < 0.0) n0 = 0.0;
    else {
        t0 *= t0;
        n0 = t0 * t0 * noise_sGrad4(noise_permutation[ii + noise_permutation[jj + noise_permutation[kk + noise_permutation[ll]]]], x0, y0, z0, w0);
    }

    float t1 = 0.6f - x1 * x1 - y1 * y1 - z1 * z1 - w1 * w1;
    if (t1 < 0.0) n1 = 0.0;
    else {
        t1 *= t1;
        n1 = t1 * t1 * noise_sGrad4(noise_permutation[ii + i1 + noise_permutation[jj + j1 + noise_permutation[kk + k1 + noise_permutation[ll + l1]]]], x1, y1, z1, w1);
    }

    float t2 = 0.6f - x2 * x2 - y2 * y2 - z2 * z2 - w2 * w2;
    if (t2 < 0.0) n2 = 0.0;
    else {
        t2 *= t2;
        n2 = t2 * t2 * noise_sGrad4(noise_permutation[ii + i2 + noise_permutation[jj + j2 + noise_permutation[kk + k2 + noise_permutation[ll + l2]]]], x2, y2, z2, w2);
    }

    float t3 = 0.6f - x3 * x3 - y3 * y3 - z3 * z3 - w3 * w3;
    if (t3 < 0.0) n3 = 0.0;
    else {
        t3 *= t3;
        n3 = t3 * t3 * noise_sGrad4(noise_permutation[ii + i3 + noise_permutation[jj + j3 + noise_permutation[kk + k3 + noise_permutation[ll + l3]]]], x3, y3, z3, w3);
    }

    float t4 = 0.6f - x4 * x4 - y4 * y4 - z4 * z4 - w4 * w4;
    if (t4 < 0.0) n4 = 0.0;
    else {
        t4 *= t4;
        n4 = t4 * t4 * noise_sGrad4(noise_permutation[ii + 1 + noise_permutation[jj + 1 + noise_permutation[kk + 1 + noise_permutation[ll + 1]]]], x4, y4, z4, w4);
    }

    // Sum up and scale the result to cover the range [-1,1]
    return 27.0f * (n0 + n1 + n2 + n3 + n4);
}

//
// ע�⣺Ҫ���� noise ���Ա���Ϊ pos ��������ԣ����磺
//#define snoise(P) (2*noise(P) - 1) // noise() function in RenderMan shading language has range [0,1]
//float DistNoise(point Pt, float distortion)
//{
//    point offset = snoise(Pt + point(0.5, 0.5, 0.5));
//    return snoise(Pt + distortion * offset);
//}
//
struct erode_noise_simplex : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3") terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float") terrain->add_attr<float>(attrName);
        }

        auto posLikeAttrName = get_input<StringObject>("posLikeAttrName")->get();
        if (!terrain->verts.has_attr(posLikeAttrName))
        {
            zeno::log_error("no such data named '{}'.", posLikeAttrName);
        }
        auto& pos = terrain->verts.attr<vec3f>(posLikeAttrName);

        terrain->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++)
            {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    float x = noise_simplexNoise3(pos[i][0], pos[i][1], pos[i][2]);
                    float y = noise_simplexNoise3(pos[i][1], pos[i][2], pos[i][0]);
                    float z = noise_simplexNoise3(pos[i][2], pos[i][0], pos[i][1]);
                    arr[i] = vec3f(x, y, z);
                }
                else
                {
                    arr[i] = noise_simplexNoise3(pos[i][0], pos[i][1], pos[i][2]);
                }
            }
            });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_noise_simplex,
    { /* inputs: */ {
            "prim_2DGrid",
            {"string", "posLikeAttrName", "pos"},
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
            {"string", "attrName", "noise"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Analytic Simplex Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// Modulo 289, optimizes to code without divisions
glm::vec3 mod289(glm::vec3 x)
{
    glm::vec3 ret{};
    ret.x = x.x - floor(x.x * (1.0 / 289.0)) * 289.0;
    ret.y = x.y - floor(x.y * (1.0 / 289.0)) * 289.0;
    ret.z = x.z - floor(x.z * (1.0 / 289.0)) * 289.0;
    return ret;
}

double mod289(double x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

// Permutation polynomial (ring size 289 = 17*17)
glm::vec3 permute(glm::vec3 x)
{
    glm::vec3 ret{};
    ret.x = mod289(((x.x * 34.0) + 10.0) * x.x);
    ret.y = mod289(((x.y * 34.0) + 10.0) * x.y);
    ret.z = mod289(((x.z * 34.0) + 10.0) * x.z);
    return ret;
}

double permute(double x) {
    return mod289(((x * 34.0) + 10.0) * x);
}

// Hashed 2-D gradients with an extra rotation.
// (The constant 0.0243902439 is 1/41)
glm::vec2 rgrad2(glm::vec2 p, double rot) {
#if 1
    // Map from a line to a diamond such that a shift maps to a rotation.
    double u = permute(permute(p.x) + p.y) * 0.0243902439 + rot; // Rotate by shift
    u = 4.0 * fract(u) - 2.0;
    // (This vector could be normalized, exactly or approximately.)
    return glm::vec2(abs(u) - 1.0, abs(abs(u + 1.0) - 2.0) - 1.0);
#else
    // For more isotropic gradients, sin/cos can be used instead.
    double u = permute(permute(p.x) + p.y) * 0.0243902439 + rot; // Rotate by shift
    u = fract(u) * 2 * M_PI;
    return glm::vec2(cos(u), sin(u));
#endif
}

//
// 2-D non-tiling simplex noise with rotating gradients and analytical derivative.
// The first component of the 3-element return vector is the noise value,
// and the second and third components are the x and y partial derivatives.
//
glm::vec3 srdnoise(glm::vec2 pos, double rot) {
    // Offset y slightly to hide some rare artifacts
    pos.y += 0.001;
    // Skew to hexagonal grid
    glm::vec2 uv = glm::vec2(pos.x + pos.y * 0.5, pos.y);

    glm::vec2 i0 = floor(uv);
    glm::vec2 f0 = fract(uv);
    // Traversal order
    glm::vec2 i1 = (f0.x > f0.y) ? glm::vec2(1.0, 0.0) : glm::vec2(0.0, 1.0);

    // Unskewed grid points in (x,y) space
    glm::vec2 p0 = glm::vec2(i0.x - i0.y * 0.5, i0.y);
    glm::vec2 p1 = glm::vec2(p0.x + i1.x - i1.y * 0.5, p0.y + i1.y);
    glm::vec2 p2 = glm::vec2(p0.x + 0.5, p0.y + 1.0);

    // Integer grid point indices in (u,v) space
    i1 = i0 + i1;
    glm::vec2 i2 = i0 + glm::vec2(1.0, 1.0);

    // Vectors in unskewed (x,y) coordinates from
    // each of the simplex corners to the evaluation point
    glm::vec2 d0 = pos - p0;
    glm::vec2 d1 = pos - p1;
    glm::vec2 d2 = pos - p2;

    glm::vec3 x = glm::vec3(p0.x, p1.x, p2.x);
    glm::vec3 y = glm::vec3(p0.y, p1.y, p2.y);
    glm::vec3 iuw = x + glm::vec3(0.5, 0.5, 0.5) * y;
    glm::vec3 ivw = y;

    // Avoid precision issues in permutation
    iuw = mod289(iuw);
    ivw = mod289(ivw);

    // Create gradients from indices
    glm::vec2 g0 = rgrad2(glm::vec2(iuw.x, ivw.x), rot);
    glm::vec2 g1 = rgrad2(glm::vec2(iuw.y, ivw.y), rot);
    glm::vec2 g2 = rgrad2(glm::vec2(iuw.z, ivw.z), rot);

    // Gradients dot vectors to corresponding corners
    // (The derivatives of this are simply the gradients)
    glm::vec3 w = glm::vec3(dot(g0, d0), dot(g1, d1), dot(g2, d2));

    // Radial weights from corners
    // 0.8 is the square of 2/sqrt(5), the distance from
    // a grid point to the nearest simplex boundary
    glm::vec3 t = glm::vec3(0.8, 0.8, 0.8) - glm::vec3(dot(d0, d0), dot(d1, d1), dot(d2, d2));

    // Partial derivatives for analytical gradient computation
    glm::vec3 dtdx = glm::vec3(-2.0, -2.0, -2.0) * glm::vec3(d0.x, d1.x, d2.x);
    glm::vec3 dtdy = glm::vec3(-2.0, -2.0, -2.0) * glm::vec3(d0.y, d1.y, d2.y);

    // Set influence of each surflet to zero outside radius sqrt(0.8)
    if (t.x < 0.0) {
        dtdx.x = 0.0;
        dtdy.x = 0.0;
        t.x = 0.0;
    }
    if (t.y < 0.0) {
        dtdx.y = 0.0;
        dtdy.y = 0.0;
        t.y = 0.0;
    }
    if (t.z < 0.0) {
        dtdx.z = 0.0;
        dtdy.z = 0.0;
        t.z = 0.0;
    }

    // Fourth power of t (and third power for derivative)
    glm::vec3 t2 = t * t;
    glm::vec3 t4 = t2 * t2;
    glm::vec3 t3 = t2 * t;

    // Final noise value is:
    // sum of ((radial weights) times (gradient dot vector from corner))
    float n = dot(t4, w);

    // Final analytical derivative (gradient of a sum of scalar products)
    glm::vec2 dt0 = glm::vec2(dtdx.x, dtdy.x) * glm::vec2(4.0, 4.0) * t3.x;
    glm::vec2 dn0 = t4.x * g0 + dt0 * w.x;
    glm::vec2 dt1 = glm::vec2(dtdx.y, dtdy.y) * glm::vec2(4.0, 4.0) * t3.y;
    glm::vec2 dn1 = t4.y * g1 + dt1 * w.y;
    glm::vec2 dt2 = glm::vec2(dtdx.z, dtdy.z) * glm::vec2(4.0, 4.0) * t3.z;
    glm::vec2 dn2 = t4.z * g2 + dt2 * w.z;

    return glm::vec3(11.0, 11.0, 11.0) * glm::vec3(n, dn0 + dn1 + dn2);
}

//
// 2-D non-tiling simplex noise with fixed gradients and analytical derivative.
// This function is implemented as a wrapper to "srdnoise",
// at the minimal cost of three extra additions.
//
glm::vec3 sdnoise(glm::vec2 pos) {
    return srdnoise(pos, 0.0);
}

struct erode_noise_analytic_simplex_2d : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto attrName = get_param<std::string>("attrName");
        if (!terrain->has_attr(attrName)) {
            terrain->add_attr<vec3f>(attrName);
        }
        auto& noise = terrain->verts.attr<vec3f>(attrName);

        auto posLikeAttrName = get_input<StringObject>("posLikeAttrName")->get();
        if (!terrain->verts.has_attr(posLikeAttrName))
        {
            zeno::log_error("no such data named '{}'.", posLikeAttrName);
        }
        auto& pos = terrain->verts.attr<vec3f>(posLikeAttrName);

        glm::vec3 ret{};// = glm::vec3(0, 0, 0);
//#pragma omp parallel for
        for (int i = 0; i < terrain->verts.size(); i++)
        {
            ret = sdnoise(glm::vec2(pos[i][0], pos[i][2]));
            noise[i] = vec3f(ret.x, ret.y, ret.z);
        }

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_noise_analytic_simplex_2d,
    { /* inputs: */ {
            "prim_2DGrid",
            {"string", "posLikeAttrName", "pos"},
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
            {"string", "attrName", "analyticNoise"},
        }, /* category: */ {
            "erode",
        } });

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Sparse Convolution Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

std::array<int, 256> perm = {
    225, 155, 210, 108, 175, 199, 221, 144, 203, 116, 70,  213, 69,  158, 33,  252, 5,   82,  173, 133, 222, 139,
    174, 27,  9,   71,  90,  246, 75,  130, 91,  191, 169, 138, 2,   151, 194, 235, 81,  7,   25,  113, 228, 159,
    205, 253, 134, 142, 248, 65,  224, 217, 22,  121, 229, 63,  89,  103, 96,  104, 156, 17,  201, 129, 36,  8,
    165, 110, 237, 117, 231, 56,  132, 211, 152, 20,  181, 111, 239, 218, 170, 163, 51,  172, 157, 47,  80,  212,
    176, 250, 87,  49,  99,  242, 136, 189, 162, 115, 44,  43,  124, 94,  150, 16,  141, 247, 32,  10,  198, 223,
    255, 72,  53,  131, 84,  57,  220, 197, 58,  50,  208, 11,  241, 28,  3,   192, 62,  202, 18,  215, 153, 24,
    76,  41,  15,  179, 39,  46,  55,  6,   128, 167, 23,  188, 106, 34,  187, 140, 164, 73,  112, 182, 244, 195,
    227, 13,  35,  77,  196, 185, 26,  200, 226, 119, 31,  123, 168, 125, 249, 68,  183, 230, 177, 135, 160, 180,
    12,  1,   243, 148, 102, 166, 38,  238, 251, 37,  240, 126, 64,  74,  161, 40,  184, 149, 171, 178, 101, 66,
    29,  59,  146, 61,  254, 107, 42,  86,  154, 4,   236, 232, 120, 21,  233, 209, 45,  98,  193, 114, 78,  19,
    206, 14,  118, 127, 48,  79,  147, 85,  30,  207, 219, 54,  88,  234, 190, 122, 95,  67,  143, 109, 137, 214,
    145, 93,  92,  100, 245, 0,   216, 186, 60,  83,  105, 97,  204, 52};

template <typename T>
constexpr T PERM(T x) {
    return perm[(x)&255];
}

#define INDEX(ix, iy, iz) PERM((ix) + PERM((iy) + PERM(iz)))

std::random_device rd;
std::default_random_engine engine(rd());
std::uniform_real_distribution<float> d(0, 1);

float impulseTab[256 * 4];
void impulseTabInit() {
    int i;
    float *f = impulseTab;
    for (i = 0; i < 256; i++) {
        *f++ = d(engine);
        *f++ = d(engine);
        *f++ = d(engine);
        *f++ = 1. - 2. * d(engine);
    }
}

float catrom2(float d, int griddist) {
    float x;
    int i;
    static float table[401];
    static bool initialized = 0;
    if (d >= griddist * griddist)
        return 0;
    if (!initialized) {
        for (i = 0; i < 4 * 100 + 1; i++) {
            x = i / (float)100;
            x = sqrtf(x);
            if (x < 1)
                table[i] = 0.5 * (2 + x * x * (-5 + x * 3));
            else
                table[i] = 0.5 * (4 + x * (-8 + x * (5 - x)));
        }
        initialized = 1;
    }
    d = d * 100 + 0.5;
    i = floor(d);
    if (i >= 4 * 100 + 1)
        return 0;
    return table[i];
}

#define NEXT(h) (((h) + 1) & 255)

float scnoise(float x, float y, float z, int pulsenum, int griddist) {
    static int initialized;
    float *fp = nullptr;
    int i, j, k, h, n;
    int ix, iy, iz;
    float sum = 0;
    float fx, fy, fz, dx, dy, dz, distsq;

    /* Initialize the random impulse table if necessary. */
    if (!initialized) {
        impulseTabInit();
        initialized = 1;
    }
    ix = floor(x);
    fx = x - ix;
    iy = floor(y);
    fy = y - iy;
    iz = floor(z);
    fz = z - iz;

    /* Perform the sparse convolution. */
    for (i = -griddist; i <= griddist; i++) { //周围的grid ： 2*griddist+1
        for (j = -griddist; j <= griddist; j++) {
            for (k = -griddist; k <= griddist; k++) {         /* Compute voxel hash code. */
                h = INDEX(ix + i, iy + j, iz + k);            //PSN
                for (n = pulsenum; n > 0; n--, h = NEXT(h)) { /* Convolve filter and impulse. */
                                                              //每个cell内随机产生pulsenum个impulse
                    fp = &impulseTab[h * 4];                  // get impulse
                    dx = fx - (i + *fp++);                    //i + *fp++   周围几个晶胞的脉冲
                    dy = fy - (j + *fp++);
                    dz = fz - (k + *fp++);
                    distsq = dx * dx + dy * dy + dz * dz;
                    sum += catrom2(distsq, griddist) *
                           *fp; // 第四个fp 指向的就是每个点的权重    filter kernel在gabor noise里面变成了gabor kernel。
                }
            }
        }
    }
    return sum / pulsenum;
}

struct erode_noise_sparse_convolution : INode {
    void apply() override {

        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        auto griddist = get_input2<int>("griddist");
        auto pulsenum = get_input2<int>("pulsenum");
        auto attrName = get_input2<std::string>("attrName:");
        auto attrType = get_input2<std::string>("attrType:");

        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3")
                terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float")
                terrain->add_attr<float>(attrName);
        }

        auto posLikeAttrName = get_input<StringObject>("posLikeAttrName")->get();
        if (!terrain->verts.has_attr(posLikeAttrName)) {
            zeno::log_error("no such data named '{}'.", posLikeAttrName);
        }

        auto &pos = terrain->verts.attr<vec3f>(posLikeAttrName);

        terrain->attr_visit(attrName, [&](auto &arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    float x = scnoise(pos[i][0], pos[i][1], pos[i][2], pulsenum, griddist);
                    float y = scnoise(pos[i][1], pos[i][2], pos[i][0], pulsenum, griddist);
                    float z = scnoise(pos[i][2], pos[i][0], pos[i][1], pulsenum, griddist);
                    arr[i] = vec3f(x, y, z);
                } else {
                    arr[i] = scnoise(pos[i][0], pos[i][1], pos[i][2], pulsenum, griddist);
                }
            }
        });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_noise_sparse_convolution, {/* inputs: */ {
                                                "prim_2DGrid",
                                                {"string", "posLikeAttrName", "pos"},
                                                {"int", "pulsenum", "3"},
                                                {"int", "griddist", "2"},
                                            },
                                            /* outputs: */
                                            {
                                                "prim_2DGrid",
                                            },
                                            /* params: */
                                            {
                                                {"string", "attrName", "noise"},
                                                {"enum float float3", "attrType", "float"},
                                            },
                                            /* category: */
                                            {
                                                "erode",
                                            }});

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Gabor Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
//reference :https://github.com/jijup/OpenSN
class pseudo_random_number_generator {
  public:
    void seed(unsigned s) {
        x_ = s;
    }
    unsigned operator()() {
        x_ *= 3039177861u;
        return x_;
    }
    float uniform_0_1() {
        return float(operator()()) / float(0xffffffff);
    } //unsigner int max
    float uniform(float min, float max) {
        return min + (uniform_0_1() * (max - min));
    }
    unsigned poisson(float mean) {
        float g_ = std::exp(-mean);
        unsigned em = 0;
        double t = uniform_0_1();
        while (t > g_) {
            ++em;
            t *= uniform_0_1();
        }
        return em;
    }

  private:
    unsigned x_;
};


float gabor(float K, float a, float F_0, float omega_0, float x, float y) {
    float gaussian_envelop = K * std::exp(-M_PI * (a * a) * ((x * x) + (y * y)));
    float sinusoidal_carrier = std::cos(2.0 * M_PI * F_0 * ((x * std::cos(omega_0)) + (y * std::sin(omega_0))));
    return gaussian_envelop * sinusoidal_carrier;
}

unsigned morton(unsigned x, unsigned y) {
    unsigned z = 0;
    for (unsigned i = 0; i < (sizeof(unsigned) * 8); ++i) { //char bit-----8
        z |= ((x & (1 << i)) << i) | ((y & (1 << i)) << (i + 1));
    }
    return z;
}

class Gnoise {
  public:
    Gnoise(float K, float a, float F_0, float omega_0, float number_of_impulses_per_kernel, 
          unsigned random_offset, bool isotropic)
        : K_(K), a_(a), F_0_(F_0), omega_0_(omega_0), random_offset_(random_offset), isotropic_(isotropic)
    {
        kernel_radius_ = std::sqrt(-std::log(0.05) / M_PI) / a_;
        impulse_density_ = number_of_impulses_per_kernel / (M_PI * kernel_radius_ * kernel_radius_);
    }

    float operator()(float x, float y) const {
        x /= kernel_radius_, y /= kernel_radius_;
        float int_x = std::floor(x), int_y = std::floor(y);
        float frac_x = x - int_x, frac_y = y - int_y;
        int i = int(int_x), j = int(int_y);
        float noise = 0.0;
        for (int di = -1; di <= +1; ++di) {
            for (int dj = -1; dj <= +1; ++dj) {
                noise += cell(i + di, j + dj, frac_x - di, frac_y - dj);
            }
        }
        return noise;
    }

    float cell(int i, int j, float x, float y) const {
 
        unsigned s = morton(i, j) + random_offset_ + 1; // nonperiodic noise
        
        pseudo_random_number_generator prng;
        prng.seed(s);

        double number_of_impulses_per_cell = impulse_density_ * kernel_radius_ * kernel_radius_;
        unsigned number_of_impulses = prng.poisson(number_of_impulses_per_cell);
        float noise = 0.0;

        for (unsigned i = 0; i < number_of_impulses; ++i) {
            float x_i = prng.uniform_0_1();
            float y_i = prng.uniform_0_1();
            float w_i = prng.uniform(-1.0, +1.0);
            float omega_0_i = prng.uniform(0.0, 2.0 * M_PI);
            float x_i_x = x - x_i;
            float y_i_y = y - y_i;
            if (((x_i_x * x_i_x) + (y_i_y * y_i_y)) < 1.0) {
                if(isotropic_)
                    noise += w_i * gabor(K_, a_, F_0_, omega_0_i, x_i_x * kernel_radius_, y_i_y * kernel_radius_);
                else
                    noise += w_i * gabor(K_, a_, F_0_, omega_0_, x_i_x * kernel_radius_, y_i_y * kernel_radius_); 
            }
        }
        return noise;
    }

    float variance() const {
        float integral_gabor_filter_squared =
            ((K_ * K_) / (4.0 * a_ * a_)) * (1.0 + std::exp(-(2.0 * M_PI * F_0_ * F_0_) / (a_ * a_)));
        return impulse_density_ * (1.0 / 3.0) * integral_gabor_filter_squared;
    }

  private:
    float K_;
    float a_;
    float F_0_;
    float omega_0_;
    float kernel_radius_;
    float impulse_density_;
    unsigned random_offset_;
    bool isotropic_;
};


struct Noise_gabor_2d : INode {
    void apply() override {

        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        auto attrName = get_input2<std::string>("attrName:");

        auto a_ = get_input2<float>("a_");
        auto F_0_ = get_input2<float>("frequency");
        auto omega_0_ = get_input2<float>("Orientation");
        auto number_of_impulses_per_kernel = get_input2<int>("impulses_per_kernel");
        auto isotropic = get_input2<bool>("isotropic");
        auto random_offset = get_input2<float>("offset");

        if (!terrain->has_attr(attrName)) {
            terrain->add_attr<float>(attrName);
        }
        auto &noise = terrain->verts.attr<float>(attrName);

        auto posLikeAttrName = get_input<StringObject>("posLikeAttrName")->get();
        if (!terrain->verts.has_attr(posLikeAttrName)) {
            zeno::log_error("no such data named '{}'.", posLikeAttrName);
        }
        auto &pos = terrain->verts.attr<vec3f>(posLikeAttrName);
        
        glm::vec3 ret{};
        auto K_ = 2.5f;  // act on spectrum

        Gnoise noise_(K_, a_, F_0_, omega_0_, number_of_impulses_per_kernel, random_offset, isotropic);
        float scale = 3.0 * std::sqrt(noise_.variance());

#pragma omp parallel for
        for (int i = 0; i < terrain->verts.size(); i++) {
            float noise2dV = 0.5 + 0.5 * noise_(pos[i][0], pos[i][2]) / scale;

            noise[i] = noise2dV; //直接float？
        }

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};

ZENDEFNODE(Noise_gabor_2d, {/* inputs: */ {
                                      "prim_2DGrid",
                                      {"string", "posLikeAttrName", "pos"},
                                      {"float", "a_", "0.07"},
                                      {"float", "frequency", "0.2"},
                                      {"float", "Orientation", "0.8"},
                                      {"int", "impulses_per_kernel", "64"},
                                      {"bool", "isotropic", "0"},
                                      {"float", "offset", "15"},
                                  },
                                  /* outputs: */
                                  {
                                      "prim_2DGrid",
                                  },
                                  /* params: */
                                  {
                                      {"string", "attrName", "noise"},
                                  },
                                  /* category: */
                                  {
                                      "erode",
                                  }});

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Worley Noise
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
glm::vec3 noise_random3(glm::vec3 p) {
    glm::vec3 val = sin(glm::vec3(dot(p, glm::vec3(127.1, 311.7, 74.7)),
        dot(p, glm::vec3(269.5, 183.3, 246.1)),
        dot(p, glm::vec3(113.5, 271.9, 124.6))));
    val *= 43758.5453123;
    return fract(val);
}

float noise_mydistance(glm::vec3 a, glm::vec3 b, int t) {
    if (t == 0) {
        return length(a - b);
    }
    else if (t == 1) {
        float xx = abs(a.x - b.x);
        float yy = abs(a.y - b.y);
        float zz = abs(a.z - b.z);
        return max(max(xx, yy), zz);
    }
    else {
        float xx = abs(a.x - b.x);
        float yy = abs(a.y - b.y);
        float zz = abs(a.z - b.z);
        return xx + yy + zz;
    }
}

float noise_WorleyNoise3(float px, float py, float pz, int fType, int distType, float offsetX, float offsetY, float offsetZ, float jitter = 1) {
    glm::vec3 pos = glm::vec3(px, py, pz);
    glm::vec3 offset = glm::vec3(offsetX, offsetY, offsetZ);
    glm::vec3 i_pos = floor(pos);
    glm::vec3 f_pos = fract(pos);

    float f1 = 9e9;
    float f2 = f1;

    for (int z = -1; z <= 1; z++) {
        for (int y = -1; y <= 1; y++) {
            for (int x = -1; x <= 1; x++) {
                glm::vec3 neighbor = glm::vec3(float(x), float(y), float(z));
                glm::vec3 point = noise_random3(i_pos + neighbor);
                point = (float)0.5 + (float)0.5 * sin(offset + (float)6.2831 * point);
                point = point * jitter;
                glm::vec3 featurePoint = neighbor + point; 

                float dist = noise_mydistance(featurePoint, f_pos, distType);
                if (dist < f1) {
                    f2 = f1; f1 = dist;
                }
                else if (dist < f2) {
                    f2 = dist;
                }
            }
        }
    }

    if (fType == 0) {
        return f1;
    }
    else {
        return f2 - f1;
    }
}

struct erode_noise_worley : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");
        auto posLikeAttrName = get_input<StringObject>("posLikeAttrName")->get();
        if (!terrain->verts.has_attr(posLikeAttrName))
        {
            zeno::log_error("no such data named '{}'.", posLikeAttrName);
        }
        auto& pos = terrain->verts.attr<vec3f>(posLikeAttrName);
        auto jitter = get_input2<float>("celljitter");
        vec3f offset;
        if (!has_input("seed")) {
            std::mt19937 gen(std::random_device{}());
            std::uniform_real_distribution<float> unif(0.f, 1.f);
            offset = vec3f(unif(gen), unif(gen), unif(gen));
        }
        else {
            offset = get_input<NumericObject>("seed")->get<vec3f>();
        }

        int fType = 0;
        auto fTypeStr = get_input2<std::string>("fType");
        //        if (fTypeStr == "F1"   ) fType = 0;
        if (fTypeStr == "F2-F1") fType = 1;

        int distType = 0;
        auto distTypeStr = get_input2<std::string>("distType");
        //        if (distTypeStr == "Euclidean") distType = 0;
        if (distTypeStr == "Chebyshev") distType = 1;
        if (distTypeStr == "Manhattan") distType = 2;

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");

        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3") terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float") terrain->add_attr<float>(attrName);
        }

        terrain->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++)
            {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>)
                {
                    float x = noise_WorleyNoise3(pos[i][0], pos[i][1], pos[i][2], fType, distType, offset[0], offset[1], offset[2], jitter);
                    float y = noise_WorleyNoise3(pos[i][1], pos[i][2], pos[i][0], fType, distType, offset[0], offset[1], offset[2], jitter);
                    float z = noise_WorleyNoise3(pos[i][2], pos[i][0], pos[i][1], fType, distType, offset[0], offset[1], offset[2], jitter);
                    arr[i] = vec3f(x, y, z);
                }
                else
                {
                    arr[i] = noise_WorleyNoise3(pos[i][0], pos[i][1], pos[i][2], fType, distType, offset[0], offset[1], offset[2], jitter);
                }
            }
            });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_noise_worley,
    { /* inputs: */ {
        "prim_2DGrid",
        "seed",
        {"string", "posLikeAttrName", "pos"},
        {"float", "celljitter", "1"},
        {"enum Euclidean Chebyshev Manhattan", "distType", "Euclidean"},
        {"enum F1 F2-F1", "fType", "F1"},
    }, /* outputs: */ {
        "prim_2DGrid",
    }, /* params: */ {
        {"string", "attrName", "noise"},
        {"enum float float3", "attrType", "float"},
    }, /* category: */ {
        "erode",
    } });


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// fractal
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
double noise_hybridMultifractal_v1(vec3f point, double H, double lacunarity, double octaves, double offset, double scale, double persistence)
{
    double frequency = 1.0;
    double amplitude = 1.0;

    double x = point[0];
    double y = point[1];
    double z = point[2];
    x *= scale;
    y *= scale;
    z *= scale;

    double result = noise_perlin(x, y, z) + offset;
    double weight = result;

    frequency *= lacunarity;
    amplitude *= persistence;

    for (int i = 1; i < octaves; i++) {
        if (weight > 1.0)
            weight = 1.0;

        double signal = (noise_perlin(x * frequency, y * frequency, z * frequency) + offset) * pow(amplitude, -H);
        result += weight * signal;
        weight *= signal;

        frequency *= lacunarity;
        amplitude *= persistence;
    }
    return result;
}

struct erode_hybridMultifractal_v1 : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto H = get_input<NumericObject>("H")->get<float>();
        auto lacunarity = get_input<NumericObject>("lacunarity")->get<float>();
        auto octaves = get_input<NumericObject>("octaves")->get<float>();
        auto offset = get_input<NumericObject>("offset")->get<float>();
        auto scale = get_input<NumericObject>("scale")->get<float>();
        auto persistence = get_input<NumericObject>("persistence")->get<float>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto& pos = terrain->verts;

        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3") terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float") terrain->add_attr<float>(attrName);
        }

        terrain->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++)
            {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>)
                {
                    float x = noise_hybridMultifractal_v1(pos[i], H, lacunarity, octaves, offset, scale, persistence);
                    arr[i] = vec3f(x, x, x);
                }
                else
                {
                    arr[i] = noise_hybridMultifractal_v1(pos[i], H, lacunarity, octaves, offset, scale, persistence);
                }
            }
            });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_hybridMultifractal_v1,
    { /* inputs: */ {
            "prim_2DGrid",
            {"float", "H", "1.0"},
            {"float", "lacunarity", "1.841"},
            {"float", "octaves", "8.0"},
            {"float", "offset", "0.8"},
            {"float", "scale", "0.002"},
            {"float", "persistence", "1.0"},
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
            {"string", "attrName", "hybrid"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });

double noise_hybridMultifractal_v2(vec3f point, double H, double lacunarity, double octaves, double offset, double scale)
{
    double x = point[0];
    double y = point[1];
    double z = point[2];
    x *= scale;
    y *= scale;
    z *= scale;

    double result = 0;
    double weight = 1;

    for (int i = 0; i < octaves; i++)
    {
        if (weight > 1.0)
            weight = 1.0;

        double signal = (noise_perlin(x, y, z) + offset) * pow(lacunarity, -H * i);
        result += weight * signal;
        weight *= signal;
        x *= lacunarity;
        y *= lacunarity;
        z *= lacunarity;
    }

    return result;
}

struct erode_hybridMultifractal_v2 : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto H = get_input<NumericObject>("H")->get<float>();
        auto lacunarity = get_input<NumericObject>("lacunarity")->get<float>();
        auto octaves = get_input<NumericObject>("octaves")->get<float>();
        auto offset = get_input<NumericObject>("offset")->get<float>();
        auto scale = get_input<NumericObject>("scale")->get<float>();
        auto persistence = get_input<NumericObject>("persistence")->get<float>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto& pos = terrain->verts;

        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3") terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float") terrain->add_attr<float>(attrName);
        }

        terrain->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++)
            {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    float x = noise_hybridMultifractal_v2(pos[i], H, lacunarity, octaves, offset, scale);// , persistence);
                    arr[i] = vec3f(x, x, x);
                }
                else {
                    arr[i] = noise_hybridMultifractal_v2(pos[i], H, lacunarity, octaves, offset, scale);// , persistence);
                }
            }
            });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_hybridMultifractal_v2,
    { /* inputs: */ {
            "prim_2DGrid",
            {"float", "H", "1.0"},
            {"float", "lacunarity", "1.841"},
            {"float", "octaves", "8.0"},
            {"float", "offset", "0.8"},
            {"float", "scale", "0.002"},
            {"float", "persistence", "1.0"},
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
            {"string", "attrName", "hybrid"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });

// blue print subnet
double noise_hybridMultifractal_v3(vec3f point, double H, double lacunarity, double octaves, double offset, double scale, double persistence)
{
    double x = point[0];
    double y = point[1];
    double z = point[2];
    x *= scale;
    y *= scale;
    z *= scale;

    double result = 0;
    double weight = 1;

    for (int i = 0; i < octaves; i++)
    {
        if (weight > 1.0)
            weight = 1.0;

        double signal = (noise_perlin(x, y, z) + offset) * pow(persistence, -H * i);
        result += weight * signal;
        weight *= signal;
        x *= lacunarity;
        y *= lacunarity;
        z *= lacunarity;
    }
    return result;
}

struct erode_hybridMultifractal_v3 : INode {
    void apply() override {
        auto terrain = get_input<PrimitiveObject>("prim_2DGrid");

        auto H = get_input<NumericObject>("H")->get<float>();
        auto lacunarity = get_input<NumericObject>("lacunarity")->get<float>();
        auto octaves = get_input<NumericObject>("octaves")->get<float>();
        auto offset = get_input<NumericObject>("offset")->get<float>();
        auto scale = get_input<NumericObject>("scale")->get<float>();
        auto persistence = get_input<NumericObject>("persistence")->get<float>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto& pos = terrain->verts;

        if (!terrain->has_attr(attrName)) {
            if (attrType == "float3") terrain->add_attr<vec3f>(attrName);
            else if (attrType == "float") terrain->add_attr<float>(attrName);
        }

        terrain->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    float x = noise_hybridMultifractal_v3(pos[i], H, lacunarity, octaves, offset, scale, persistence);
                    arr[i] = vec3f(x, x, x);
                }
                else {
                    arr[i] = noise_hybridMultifractal_v3(pos[i], H, lacunarity, octaves, offset, scale, persistence);
                }
            }
            });

        set_output("prim_2DGrid", get_input("prim_2DGrid"));
    }
};
ZENDEFNODE(erode_hybridMultifractal_v3,
    { /* inputs: */ {
            "prim_2DGrid",
            {"float", "H", "1.0"},
            {"float", "lacunarity", "1.841"},
            {"float", "octaves", "8.0"},
            {"float", "offset", "0.8"},
            {"float", "scale", "0.002"},
            {"float", "persistence", "1.0"},
        }, /* outputs: */ {
            "prim_2DGrid",
        }, /* params: */ {
            {"string", "attrName", "hybrid"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Domain Warping
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
float noise_fbm(vec3f pos, float H, float lacunarity, float frequence, float amplitude, int Octaves)
{
    //float G = exp2(-H);    // float G = 0.5;
    float t = 0.0;
    for (int i = 0; i < Octaves; i++)
    {
        amplitude = pow(lacunarity, -H * i);
        t += amplitude * noise_perlin(frequence * pos[0], frequence * pos[1], frequence * pos[2]);
        //frequence *= 2.0;
        frequence *= lacunarity;
        //amplitude *= G;
    }
    return t;
}

float noise_domainWarpingV1(vec3f pos, float H, float frequence, float amplitude, int numOctaves)
{
    vec3f q = vec3f(noise_fbm(pos + vec3f(0.0, 0.0, 0.0), H, 2.0f, frequence, amplitude, numOctaves),
        noise_fbm(pos + vec3f(1.7, 2.8, 9.2), H, 2.0f, frequence, amplitude, numOctaves),
        noise_fbm(pos + vec3f(5.2, 8.3, 1.3), H, 2.0f, frequence, amplitude, numOctaves));
    return noise_fbm(pos + 4.0 * q, H, 2.0f, frequence, amplitude, numOctaves);
}

struct erode_domainWarping_v1 : INode {
    void apply() override {
        auto prim = has_input("prim") ? get_input<PrimitiveObject>("prim") : std::make_shared<PrimitiveObject>();

        auto H = get_input<NumericObject>("fbmH")->get<float>();
        auto frequence = get_input<NumericObject>("fbmFrequence")->get<float>();
        auto amplitude = get_input<NumericObject>("fbmAmplitude")->get<float>();
        auto numOctaves = get_input<NumericObject>("fbmNumOctaves")->get<int>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto& pos = prim->verts;
        if (!prim->has_attr(attrName)) {
            if (attrType == "float3") prim->add_attr<vec3f>(attrName);
            else if (attrType == "float") prim->add_attr<float>(attrName);
        }

        prim->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    float x = noise_domainWarpingV1(pos[i], H, frequence, amplitude, numOctaves);
                    float y = noise_domainWarpingV1(pos[i], H, frequence, amplitude, numOctaves);
                    float z = noise_domainWarpingV1(pos[i], H, frequence, amplitude, numOctaves);
                    arr[i] = vec3f(x, y, z);
                }
                else {
                    arr[i] = noise_domainWarpingV1(pos[i], H, frequence, amplitude, numOctaves);
                }
            }
            });

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(erode_domainWarping_v1,
    { /* inputs: */ {
            "prim",
            {"float", "fbmH", "1.0"},
            {"float", "fbmFrequence", "1.0"},
            {"float", "fbmAmplitude", "1.0"},
            {"int", "fbmNumOctaves", "4"},
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {
            {"string", "attrName", "noise"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });

float noise_domainWarpingV2(vec3f pos, float H, float frequence, float amplitude, int numOctaves)
{
    vec3f q = vec3f(noise_fbm(pos + vec3f(0.0, 0.0, 0.0), H, 2.0f, frequence, amplitude, numOctaves),
        noise_fbm(pos + vec3f(1.7, 2.8, 9.2), H, 2.0f, frequence, amplitude, numOctaves),
        noise_fbm(pos + vec3f(5.2, 8.3, 1.3), H, 2.0f, frequence, amplitude, numOctaves));

    vec3f r = vec3f(noise_fbm(pos + 4.0 * q + vec3f(2.8, 9.2, 1.7), H, 2.0f, frequence, amplitude, numOctaves),
        noise_fbm(pos + 4.0 * q + vec3f(9.2, 1.7, 2.8), H, 2.0f, frequence, amplitude, numOctaves),
        noise_fbm(pos + 4.0 * q + vec3f(1.3, 5.2, 8.3), H, 2.0f, frequence, amplitude, numOctaves));

    return noise_fbm(pos + 4.0 * r, H, 2.0f, frequence, amplitude, numOctaves);
}

struct erode_domainWarping_v2 : INode {
    void apply() override {
        auto prim = has_input("prim") ?
            get_input<PrimitiveObject>("prim") :
            std::make_shared<PrimitiveObject>();

        auto H = get_input<NumericObject>("fbmH")->get<float>();
        auto frequence = get_input<NumericObject>("fbmFrequence")->get<float>();
        auto amplitude = get_input<NumericObject>("fbmAmplitude")->get<float>();
        auto numOctaves = get_input<NumericObject>("fbmNumOctaves")->get<int>();

        auto attrName = get_param<std::string>("attrName");
        auto attrType = get_param<std::string>("attrType");
        auto& pos = prim->verts;
        if (!prim->has_attr(attrName)) {
            if (attrType == "float3") prim->add_attr<vec3f>(attrName);
            else if (attrType == "float") prim->add_attr<float>(attrName);
        }

        prim->attr_visit(attrName, [&](auto& arr) {
#pragma omp parallel for
            for (int i = 0; i < arr.size(); i++) {
                if constexpr (is_decay_same_v<decltype(arr[i]), vec3f>) {
                    float x = noise_domainWarpingV2(pos[i], H, frequence, amplitude, numOctaves);
                    float y = noise_domainWarpingV2(pos[i], H, frequence, amplitude, numOctaves);
                    float z = noise_domainWarpingV2(pos[i], H, frequence, amplitude, numOctaves);
                    arr[i] = vec3f(x, y, z);
                }
                else {
                    arr[i] = noise_domainWarpingV2(pos[i], H, frequence, amplitude, numOctaves);
                }
            }
            });

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(erode_domainWarping_v2,
    { /* inputs: */ {
            "prim",
            {"float", "fbmH", "1.0"},
            {"float", "fbmFrequence", "1.0"},
            {"float", "fbmAmplitude", "1.0"},
            {"int", "fbmNumOctaves", "4"},
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {
            {"string", "attrName", "noise"},
            {"enum float float3", "attrType", "float"},
        }, /* category: */ {
            "erode",
        } });


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Voronoi
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
void noise_Voronoi3(const vec3f pos, const std::vector<vec3f>& points, float& voronoi, vec3f& minPoint) {
    float minDist = 9e9;
    for (auto const& point : points)
    {
        float dist = length(pos - point);
        if (dist < minDist) {
            minDist = dist;
            minPoint = point;
        }
    }
    voronoi = minDist;
}

struct erode_voronoi : INode {
    void apply() override {
        auto prim = has_input("prim") ? get_input<PrimitiveObject>("prim") : std::make_shared<PrimitiveObject>();
        auto featurePrim = has_input("featurePrim") ? get_input<PrimitiveObject>("featurePrim") : std::make_shared<PrimitiveObject>();

        auto attrName = get_param<std::string>("attrName");
        if (!prim->has_attr(attrName)) { prim->add_attr<float>(attrName); }
        if (!prim->has_attr("minFeaturePointPos")) { prim->add_attr<vec3f>("minFeaturePointPos"); }

        auto& attr_voro = prim->attr<float>(attrName);
        auto& attr_mFPP = prim->attr<vec3f>("minFeaturePointPos");

        auto& samplePoints = prim->verts;
        auto& featurePoints = featurePrim->verts;
#pragma omp parallel for
        for (int i = 0; i < prim->size(); i++) {
            noise_Voronoi3(samplePoints[i], featurePrim->verts, attr_voro[i], attr_mFPP[i]);
        }

        set_output("prim", get_input("prim"));
    }
};
ZENDEFNODE(erode_voronoi,
    { /* inputs: */ {
            "prim",
            "featurePrim",
        }, /* outputs: */ {
            "prim",
        }, /* params: */ {
            {"string", "attrName", "voronoi"},
        }, /* category: */ {
            "erode",
        } });






} // namespace
} // namespace zeno