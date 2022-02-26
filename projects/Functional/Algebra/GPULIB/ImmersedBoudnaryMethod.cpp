/**
 * @file ImmersedBoundaryMethod.cpp
 * @author Ma Pengfei (mapengfei@mail.nwpu.edu.cn)
 * @brief 
 * @version 0.1
 * @date 2021-12-07
 * 
 * @copyright Copyright (c) 2021  Ma Pengfei
 * 
 */
#include <helper_cuda.h>
#include <helper_math.h>

namespace gpu{

double ibm_phi(double r)
{
    double res=0;
    double r2 = r*r;
    if(r<=2)
        res = 0.125*(5-2*r-sqrt(-7+12*r-4*r2));
    if(r<=1)
        res = 0.125*(3-2*r+sqrt(1+4*r-4*r2));
    return res;
}

double ibm_delta3(double3 x, double3 g)
{

    return ibm_phi(abs(x.x-g.x))*ibm_phi(abs(x.y-g.y))*ibm_phi(abs(x.z-g.z));
}

void distribute_force_cpu(
    const double3* solid_forces,                        /// solid_forces
    const double4* quadrature_rules,                    /// quadrature quadrature_rules
          double3* fluid_forces,
    int num, double h, int3 dim)
{
    for (size_t gidx = 0; gidx < num; gidx++)
    {
        double4 quadrature_rule = quadrature_rules[gidx];
        int i = floor(quadrature_rule.x/h);
        int j = floor(quadrature_rule.y/h);
        int k = floor(quadrature_rule.z/h);
        double w = quadrature_rule.w;
        double3 f = solid_forces[gidx];
        double inv_h3 = 1.0/h/h/h;

        for(int kk=k-1;kk<=k+2;kk++)
        {
            for(int jj=j-1;jj<=j+2;jj++)
            {
                for(int ii=i-1;ii<=i+2;ii++)
                {
                    if(!(kk>=0&&kk<dim.z&&jj>=0&&jj<dim.y&&ii>=0&&ii<dim.x))
                    {
                        continue;
                    }
                    int widx = ii + jj*dim.x + kk*dim.x*dim.y;
                    double3 spreadf;
                    double weight = w*inv_h3*ibm_delta3(make_double3(quadrature_rule.x/h, quadrature_rule.y/h, quadrature_rule.z/h), make_double3(ii,jj,kk));
                    spreadf.x = f.x*weight;
                    spreadf.y = f.y*weight;
                    spreadf.z = f.z*weight;
                    fluid_forces[widx].x += spreadf.x;
                    fluid_forces[widx].y += spreadf.y;
                    fluid_forces[widx].z += spreadf.z;
                }
            }
        }
    }
}


void interpolate_velocity_cpu(
          double3* solid_velocities, 
    const double4* quadrature_rules, 
    const double3* fluid_velocities, 
    int num, double h, int3 dim)
{
    for (size_t gidx = 0; gidx < num; gidx++){
        double4 pos = quadrature_rules[gidx];
        int i = floor(pos.x/h);
        int j = floor(pos.y/h);
        int k = floor(pos.z/h);
        double3 sum = make_double3(0,0,0);
        for (int kk = k - 1; kk <= k + 2; kk++) {
            for (int jj = j - 1; jj <= j + 2; jj++) {
                for (int ii = i - 1; ii <= i + 2; ii++) {
                    if (!(kk >= 0 && kk < dim.z && jj >= 0 && jj < dim.y && ii >= 0 && ii < dim.x)) {
                        continue;
                    }
                    // NOTE : which one is correct?
                    // int ridx = ii + jj * dim.x + kk * dim.x * dim.y;
                    int ridx = ii + jj * dim.x + kk * dim.x * dim.y;
                    double weight = ibm_delta3(make_double3(pos.x / h, pos.y / h, pos.z / h),
                                               make_double3(ii, jj, kk));
                    // BUG : __syncthreads() here is a bug.
                    // __syncthreads();
                    double3 gvalue = fluid_velocities[ridx];
                    sum.x += weight * gvalue.x;
                    sum.y += weight * gvalue.y;
                    sum.z += weight * gvalue.z;
                    // printf("interpolate_velocity_kernel : %d, %lf, %lf, %lf\n", ridx, gvalue.x, gvalue.y, gvalue.z);
                }
            }
        }
        solid_velocities[gidx] = sum;
    }
}
}