#include "marching_tiles_hires.h"


namespace LosTopos {

// definition of acute tile
static const int num_nodes=40;
static Vec3i node[num_nodes]={
   Vec3i(1,0,0),
   Vec3i(2,2,0),
   Vec3i(1,4,0),
   Vec3i(3,4,0),
   Vec3i(2,6,0),
   Vec3i(1,0,4),
   Vec3i(3,0,4),
   Vec3i(2,1,2),
   Vec3i(0,2,1),
   Vec3i(0,2,3),
   Vec3i(2,2,4),   // 10
   Vec3i(1,4,4),
   Vec3i(3,4,4),
   Vec3i(0,4,2),
   Vec3i(2,3,2),
   Vec3i(2,5,2),
   Vec3i(4,2,1),
   Vec3i(4,2,3),
   Vec3i(5,4,4),
   Vec3i(4,4,2),
   Vec3i(0,6,1),   // 20
   Vec3i(0,6,3),
   Vec3i(2,6,4),
   Vec3i(2,7,2),
   Vec3i(4,6,1),
   Vec3i(4,6,3),
   Vec3i(0,2,5),
   Vec3i(2,3,6),
   Vec3i(2,5,6),
   Vec3i(4,2,5),
   Vec3i(4,4,6),   // 30
   Vec3i(4,6,5),
   Vec3i(2,1,6),
   Vec3i(0,0,2),
   Vec3i(5,0,4),
   Vec3i(4,0,2),
   Vec3i(3,0,0),
   Vec3i(4,0,6),
   Vec3i(5,0,0),
   Vec3i(5,4,0)
};
static const int num_tets=46;
static Vec4i tet[num_tets]={
   Vec4i(2,3,15,14),
   Vec4i(2,15,13,14),
   Vec4i(6,29,17,10),
   Vec4i(6,17,34,35),
   Vec4i(12,14,17,10),
   Vec4i(0,36,1,7),
   Vec4i(29,12,18,17),
   Vec4i(3,14,16,19),
   Vec4i(3,15,14,19),
   Vec4i(14,16,1,3),
   Vec4i(0,7,8,33),
   Vec4i(7,16,36,1),
   Vec4i(9,7,5,33),
   Vec4i(8,7,9,33),
   Vec4i(14,12,17,19),
   Vec4i(12,29,10,17),
   Vec4i(14,9,13,8),
   Vec4i(8,2,14,1),
   Vec4i(14,17,16,19),
   Vec4i(17,14,7,10),
   Vec4i(16,7,36,35),
   Vec4i(17,6,7,35),
   Vec4i(11,15,14,13),
   Vec4i(3,2,1,14),
   Vec4i(9,14,7,8),
   Vec4i(9,14,11,10),
   Vec4i(1,8,0,7),
   Vec4i(14,8,1,7),
   Vec4i(12,15,19,14),
   Vec4i(19,39,16,3),
   Vec4i(17,12,18,19),
   Vec4i(14,9,11,13),
   Vec4i(14,12,11,10),
   Vec4i(2,8,14,13),
   Vec4i(6,17,7,10),
   Vec4i(5,9,26,10),
   Vec4i(14,9,7,10),
   Vec4i(11,9,10,26),
   Vec4i(7,9,5,10),
   Vec4i(17,6,34,29),
   Vec4i(6,7,5,10),
   Vec4i(15,12,11,14),
   Vec4i(16,14,1,7),
   Vec4i(7,17,35,16),
   Vec4i(38,35,16,36),
   Vec4i(14,16,17,7)
};

void MarchingTilesHiRes::
contour(void)
{
   tri.resize(0);
   x.resize(0);
   edge_cross.clear();
   for(int k=0; k<phi.nk; ++k) for(int j=0; j<phi.nj; ++j) for(int i=0; i<phi.ni; ++i)
      contour_tile(i,j,k);
}

void MarchingTilesHiRes::
improve_mesh(void)
{
   // first get adjacency information
   std::vector<Array1ui> nbr(x.size());
   for(unsigned int t=0; t<tri.size(); ++t){
      size_t p, q, r; assign(tri[t], p, q, r);
      nbr[p].add_unique((unsigned int)q);
      nbr[p].add_unique((unsigned int)r);
      nbr[q].add_unique((unsigned int)p);
      nbr[q].add_unique((unsigned int)r);
      nbr[r].add_unique((unsigned int)p);
      nbr[r].add_unique((unsigned int)q);
   }
   // then sweep through the mesh a few times incrementally improving positions
   for(unsigned int sweep=0; sweep<3; ++sweep){
      for(unsigned int p=0; p<x.size(); ++p){
         // get a weighted average of neighbourhood positions
         Vec3d target=x[p];
         for(unsigned int a=0; a<nbr[p].size(); ++a)
            target+=x[nbr[p][a]];
         target/=(1.f+nbr[p].size());
         // project onto level set surface with Newton
         for(int projection_step=0; projection_step<5; ++projection_step){
            double i=(target[0]-origin[0])/dx, j=(target[1]-origin[1])/dx, k=(target[2]-origin[2])/dx;
            double f=eval(i,j,k);
            Vec3d g; eval_gradient(i,j,k,g);
            double m2=mag2(g), m=std::sqrt(m2);
            double alpha=clamp(-f/(m2+1e-30), -0.25*m, 0.25*m); // clamp to avoid stepping more than a fraction of a grid cell
            // do line search to make sure we actually are getting closer to the zero level set
            bool line_search_success=false;
            for(int line_search_step=0; line_search_step<10; ++line_search_step){
               double fnew=eval(i+alpha*g[0], j+alpha*g[1], k+alpha*g[2]);
               if(std::fabs(fnew)<=std::fabs(f)){
                  target += Vec3d( (alpha*dx)*g[0], (alpha*dx)*g[0], (alpha*dx)*g[0] );
                  target += Vec3d( (alpha*dx)*g[1], (alpha*dx)*g[1], (alpha*dx)*g[1] );
                  target += Vec3d( (alpha*dx)*g[2], (alpha*dx)*g[2], (alpha*dx)*g[2] );
                  line_search_success=true;
                  break;
               }else
                  alpha*=0.5;
            }
            if(!line_search_success){ // if we stalled trying to find the zero isocontour...
               // weight the target closer to the original x[p]
               std::cout<<"line search failed (p="<<p<<" project="<<projection_step<<" sweep="<<sweep<<")"<<std::endl;
               target= 0.5 * (x[p]+target);
            }
         }
         x[p]=target;
      }
   }
}

void MarchingTilesHiRes::
estimate_normals(void)
{
   normal.resize(x.size());
   for(unsigned int p=0; p<x.size(); ++p){
      eval_gradient((x[p][0]-origin[0])/dx,
                    (x[p][1]-origin[1])/dx,
                    (x[p][2]-origin[2])/dx,
                    normal[p]);
      normalize(normal[p]);
   }
}

double MarchingTilesHiRes::
eval(double i, double j, double k)
{
   int p, q, r;
   double f, g, h;
   /*
   get_barycentric(i, p, f, 0, phi.ni);
   get_barycentric(j, q, g, 0, phi.nj);
   get_barycentric(k, r, h, 0, phi.nk);
   return trilerp(phi(p,q,r), phi(p+1,q,r), phi(p,q+1,r), phi(p+1,q+1,r),
                  phi(p,q,r+1), phi(p+1,q,r+1), phi(p,q+1,r+1), phi(p+1,q+1,r+1), f, g, h);
   */
   get_barycentric(i+0.5f, p, f, 1, phi.ni);
   get_barycentric(j+0.5f, q, g, 1, phi.nj);
   get_barycentric(k+0.5f, r, h, 1, phi.nk);
   double wx0, wx1, wx2, wy0, wy1, wy2, wz0, wz1, wz2;
   quadratic_bspline_weights(f, wx0, wx1, wx2);
   quadratic_bspline_weights(g, wy0, wy1, wy2);
   quadratic_bspline_weights(h, wz0, wz1, wz2);
   return wx0*( wy0*( wz0*phi(p-1,q-1,r-1) + wz1*phi(p-1,q-1,r) + wz2*phi(p-1,q-1,r+1) )
               +wy1*( wz0*phi(p-1,q,  r-1) + wz1*phi(p-1,q,  r) + wz2*phi(p-1,q,  r+1) )
               +wy2*( wz0*phi(p-1,q+1,r-1) + wz1*phi(p-1,q+1,r) + wz2*phi(p-1,q+1,r+1) ) )
         +wx1*( wy0*( wz0*phi(p,  q-1,r-1) + wz1*phi(p,  q-1,r) + wz2*phi(p,  q-1,r+1) )
               +wy1*( wz0*phi(p,  q,  r-1) + wz1*phi(p,  q,  r) + wz2*phi(p,  q,  r+1) )
               +wy2*( wz0*phi(p,  q+1,r-1) + wz1*phi(p,  q+1,r) + wz2*phi(p,  q+1,r+1) ) )
         +wx2*( wy0*( wz0*phi(p+1,q-1,r-1) + wz1*phi(p+1,q-1,r) + wz2*phi(p+1,q-1,r+1) )
               +wy1*( wz0*phi(p+1,q,  r-1) + wz1*phi(p+1,q,  r) + wz2*phi(p+1,q,  r+1) )
               +wy2*( wz0*phi(p+1,q+1,r-1) + wz1*phi(p+1,q+1,r) + wz2*phi(p+1,q+1,r+1) ) );
}

void MarchingTilesHiRes::
eval_gradient(double i, double j, double k, Vec3d& grad)
{
   /*
   double gx=(eval(i+1e-3d,j,k)-eval(i-1e-3d,j,k))/2e-3d;
   double gy=(eval(i,j+1e-3d,k)-eval(i,j-1e-3d,k))/2e-3d;
   double gz=(eval(i,j,k+1e-3d)-eval(i,j,k-1e-3d))/2e-3d;
   grad[0]=gx;
   grad[1]=gy;
   grad[2]=gz;
   */
   int p, q, r;
   double f, g, h;
   get_barycentric(i+0.5f, p, f, 1, phi.ni);
   get_barycentric(j+0.5f, q, g, 1, phi.nj);
   get_barycentric(k+0.5f, r, h, 1, phi.nk);
   double wx0, wx1, wx2, wy0, wy1, wy2, wz0, wz1, wz2;
   quadratic_bspline_weights(f, wx0, wx1, wx2);
   quadratic_bspline_weights(g, wy0, wy1, wy2);
   quadratic_bspline_weights(h, wz0, wz1, wz2);

   grad[0]=wz0*( wy0*lerp(phi(p,q-1,r-1)-phi(p-1,q-1,r-1), phi(p+1,q-1,r-1)-phi(p,q-1,r-1), f)
                +wy1*lerp(phi(p,q,  r-1)-phi(p-1,q,  r-1), phi(p+1,q,  r-1)-phi(p,q,  r-1), f)
                +wy2*lerp(phi(p,q+1,r-1)-phi(p-1,q+1,r-1), phi(p+1,q+1,r-1)-phi(p,q+1,r-1), f) )
          +wz1*( wy0*lerp(phi(p,q-1,r  )-phi(p-1,q-1,r  ), phi(p+1,q-1,r  )-phi(p,q-1,r  ), f)
                +wy1*lerp(phi(p,q,  r  )-phi(p-1,q,  r  ), phi(p+1,q,  r  )-phi(p,q,  r  ), f)
                +wy2*lerp(phi(p,q+1,r  )-phi(p-1,q+1,r  ), phi(p+1,q+1,r  )-phi(p,q+1,r  ), f) )
          +wz2*( wy0*lerp(phi(p,q-1,r+1)-phi(p-1,q-1,r+1), phi(p+1,q-1,r+1)-phi(p,q-1,r+1), f)
                +wy1*lerp(phi(p,q,  r+1)-phi(p-1,q,  r+1), phi(p+1,q,  r+1)-phi(p,q,  r+1), f)
                +wy2*lerp(phi(p,q+1,r+1)-phi(p-1,q+1,r+1), phi(p+1,q+1,r+1)-phi(p,q+1,r+1), f) );

   grad[1]=wz0*( wx0*lerp(phi(p-1,q,r-1)-phi(p-1,q-1,r-1), phi(p-1,q+1,r-1)-phi(p-1,q,r-1), g)
                +wx1*lerp(phi(p,  q,r-1)-phi(p,  q-1,r-1), phi(p,  q+1,r-1)-phi(p,  q,r-1), g)
                +wx2*lerp(phi(p+1,q,r-1)-phi(p+1,q-1,r-1), phi(p+1,q+1,r-1)-phi(p+1,q,r-1), g) )
          +wz1*( wx0*lerp(phi(p-1,q,r  )-phi(p-1,q-1,r  ), phi(p-1,q+1,r  )-phi(p-1,q,r  ), g)
                +wx1*lerp(phi(p,  q,r  )-phi(p,  q-1,r  ), phi(p,  q+1,r  )-phi(p,  q,r  ), g)
                +wx2*lerp(phi(p+1,q,r  )-phi(p+1,q-1,r  ), phi(p+1,q+1,r  )-phi(p+1,q,r  ), g) )
          +wz2*( wx0*lerp(phi(p-1,q,r+1)-phi(p-1,q-1,r+1), phi(p-1,q+1,r+1)-phi(p-1,q,r+1), g)
                +wx1*lerp(phi(p,  q,r+1)-phi(p,  q-1,r+1), phi(p,  q+1,r+1)-phi(p,  q,r+1), g)
                +wx2*lerp(phi(p+1,q,r+1)-phi(p+1,q-1,r+1), phi(p+1,q+1,r+1)-phi(p+1,q,r+1), g) );

   grad[2]=wx0*( wy0*lerp(phi(p-1,q-1,r)-phi(p-1,q-1,r-1), phi(p-1,q-1,r+1)-phi(p-1,q-1,r), h)
                +wy1*lerp(phi(p-1,q,  r)-phi(p-1,q,  r-1), phi(p-1,q,  r+1)-phi(p-1,q,  r), h)
                +wy2*lerp(phi(p-1,q+1,r)-phi(p-1,q+1,r-1), phi(p-1,q+1,r+1)-phi(p-1,q+1,r), h) )
          +wx1*( wy0*lerp(phi(p,  q-1,r)-phi(p,  q-1,r-1), phi(p,  q-1,r+1)-phi(p,  q-1,r), h)
                +wy1*lerp(phi(p,  q,  r)-phi(p,  q,  r-1), phi(p,  q,  r+1)-phi(p,  q,  r), h)
                +wy2*lerp(phi(p,  q+1,r)-phi(p,  q+1,r-1), phi(p,  q+1,r+1)-phi(p,  q+1,r), h) )
          +wx2*( wy0*lerp(phi(p+1,q-1,r)-phi(p+1,q-1,r-1), phi(p+1,q-1,r+1)-phi(p+1,q-1,r), h)
                +wy1*lerp(phi(p+1,q,  r)-phi(p+1,q,  r-1), phi(p+1,q,  r+1)-phi(p+1,q,  r), h)
                +wy2*lerp(phi(p+1,q+1,r)-phi(p+1,q+1,r-1), phi(p+1,q+1,r+1)-phi(p+1,q+1,r), h) );

}

void MarchingTilesHiRes::
contour_tile(int i, int j, int k)
{
   Vec3i cell(4*i,4*j,4*k); // coordinates on the acute lattice, not the grid for phi
   for(int t=0; t<num_tets; ++t){
      int a, b, c, d; assign(tet[t], a, b, c, d);
      contour_tet(cell+node[a], cell+node[b], cell+node[c], cell+node[d],
                  eval(0.25f*(cell[0]+node[a][0]), 0.25f*(cell[1]+node[a][1]), 0.25f*(cell[2]+node[a][2])),
                  eval(0.25f*(cell[0]+node[b][0]), 0.25f*(cell[1]+node[b][1]), 0.25f*(cell[2]+node[b][2])),
                  eval(0.25f*(cell[0]+node[c][0]), 0.25f*(cell[1]+node[c][1]), 0.25f*(cell[2]+node[c][2])),
                  eval(0.25f*(cell[0]+node[d][0]), 0.25f*(cell[1]+node[d][1]), 0.25f*(cell[2]+node[d][2])));
   }
}

// contour the tet with given grid point vertices and function values
// --- corners arranged so that 0-1-2 uses right-hand-rule to get to 3
void MarchingTilesHiRes::
contour_tet(const Vec3i& x0, const Vec3i& x1, const Vec3i& x2, const Vec3i& x3, double p0, double p1, double p2, double p3)
{
   // guard against topological degeneracies
   if(p0==0) p0=1e-30f;
   if(p1==0) p1=1e-30f;
   if(p2==0) p2=1e-30f;
   if(p3==0) p3=1e-30f;

   if(p0<0){
      if(p1<0){
         if(p2<0){
            if(p3<0){
               return; // no contour here
            }else // p3>=0
               tri.push_back(Vec3st(find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x1,x3,p1,p3),
                                   find_edge_cross(x2,x3,p2,p3)));
         }else{ // p2>=0
            if(p3<0)
               tri.push_back(Vec3st(find_edge_cross(x0,x2,p0,p2),
                                   find_edge_cross(x3,x2,p3,p2),
                                   find_edge_cross(x1,x2,p1,p2)));
            else{ // p3>=0
               tri.push_back(Vec3st(find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x1,x3,p1,p3),
                                   find_edge_cross(x0,x2,p0,p2)));
               tri.push_back(Vec3st(find_edge_cross(x1,x3,p1,p3),
                                   find_edge_cross(x1,x2,p1,p2),
                                   find_edge_cross(x0,x2,p0,p2)));
            }
         }
      }else{ // p1>=0
         if(p2<0){
            if(p3<0)
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x2,x1,p2,p1),
                                   find_edge_cross(x3,x1,p3,p1)));
            else{ // p3>=0
               tri.push_back(Vec3st(find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x2,x3,p2,p3)));
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x2,x1,p2,p1),
                                   find_edge_cross(x2,x3,p2,p3)));
            }
         }else{ // p2>=0
            if(p3<0){
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x0,x2,p0,p2),
                                   find_edge_cross(x3,x2,p3,p2)));
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x3,x2,p3,p2),
                                   find_edge_cross(x3,x1,p3,p1)));
            }else // p3>=_0
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x0,x2,p0,p2),
                                   find_edge_cross(x0,x3,p0,p3)));
         }
      }
   }else{ // p0>=0
      if(p1<0){
         if(p2<0){
            if(p3<0)
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x0,x2,p0,p2)));
            else{ // p3>=0
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x3,x1,p3,p1),
                                   find_edge_cross(x3,x2,p3,p2)));
               tri.push_back(Vec3st(find_edge_cross(x3,x2,p3,p2),
                                   find_edge_cross(x0,x2,p0,p2),
                                   find_edge_cross(x0,x1,p0,p1)));
            }
         }else{ // p2>=0
            if(p3<0){
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x3,x2,p3,p2)));
               tri.push_back(Vec3st(find_edge_cross(x0,x1,p0,p1),
                                   find_edge_cross(x3,x2,p3,p2),
                                   find_edge_cross(x2,x1,p2,p1)));
            }else // p3>=0
               tri.push_back(Vec3st(find_edge_cross(x1,x0,p1,p0),
                                   find_edge_cross(x1,x3,p1,p3),
                                   find_edge_cross(x1,x2,p1,p2)));
         }
      }else{ // p1>=0
         if(p2<0){
            if(p3<0){
               tri.push_back(Vec3st(find_edge_cross(x1,x3,p1,p3),
                                   find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x0,x2,p0,p2)));
               tri.push_back(Vec3st(find_edge_cross(x1,x3,p1,p3),
                                   find_edge_cross(x0,x2,p0,p2),
                                   find_edge_cross(x1,x2,p1,p2)));
            }else // p3>=0
               tri.push_back(Vec3st(find_edge_cross(x0,x2,p0,p2),
                                   find_edge_cross(x1,x2,p1,p2),
                                   find_edge_cross(x3,x2,p3,p2)));
         }else{ // p2>=0
            if(p3<0)
               tri.push_back(Vec3st(find_edge_cross(x0,x3,p0,p3),
                                   find_edge_cross(x2,x3,p2,p3),
                                   find_edge_cross(x1,x3,p1,p3)));
            else{ // p3>=0
               return; // assume no degenerate cases (where some of the p's are zero)
            }
         }
      }
   }
}

// return the vertex of the edge crossing (create it if necessary) between given grid points and function values
int MarchingTilesHiRes::
find_edge_cross(const Vec3i& x0, const Vec3i& x1, double p0, double p1)
{
   unsigned int vertex_index;
   if(edge_cross.get_entry(Vec6i(x0.v[0], x0.v[1], x0.v[2], x1.v[0], x1.v[1], x1.v[2]), vertex_index)){
      return vertex_index;
   }else if(edge_cross.get_entry(Vec6i(x1.v[0], x1.v[1], x1.v[2], x0.v[0], x0.v[1], x0.v[2]), vertex_index)){
      return vertex_index;
   }else{
      double a=p1/(p1-p0), b=1-a;
      vertex_index=(int)x.size();
      x.push_back(Vec3d(origin[0]+dx*0.25f*(a*x0[0]+b*x1[0]),
                        origin[1]+dx*0.25f*(a*x0[1]+b*x1[1]),
                        origin[2]+dx*0.25f*(a*x0[2]+b*x1[2])));
      edge_cross.add(Vec6i(x0.v[0], x0.v[1], x0.v[2], x1.v[0], x1.v[1], x1.v[2]), vertex_index);
      return vertex_index;
   }
}

}