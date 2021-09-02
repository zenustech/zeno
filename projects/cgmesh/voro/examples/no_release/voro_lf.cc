// File import example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

#include <vector>
using namespace std;

// Set up constants for the container geometry
const double ax=-0.5,bx=25.5;
const double ay=-0.5,by=25.5;
const double az=-0.5,bz=25.5;

int main() {
	
	// Manually import the file
	int i,j,id,max_id=0,n;
	double x,y,z;
	vector<int> vid,neigh,f_order;
	vector<double> vx,vy,vz,vd;
	FILE *fp=safe_fopen("liq-900K.dat","r"),*fp2,*fp3;
	while((j=fscanf(fp,"%d %lg %lg %lg",&id,&x,&y,&z))==4) {
		vid.push_back(id);if(id>max_id) max_id=id;
		vx.push_back(x);
		vy.push_back(y);
		vz.push_back(z);
	}
	if(j!=EOF) voro_fatal_error("File import error",VOROPP_FILE_ERROR);
	n=vid.size();
	fclose(fp);

	// Compute optimal size for container, and then construct the container
	double dx=bx-ax,dy=by-ay,dz=bz-az;
	double l(pow(n/(5.6*dx*dy*dz),1/3.0));
	int nx=int(dx*l+1),ny=int(dy*l+1),nz=int(dz*l+1);
	container con(ax,bx,ay,by,az,bz,nx,ny,nz,false,false,false,8);

	// Print status message
	printf("Read %d particles, max ID is %d\n"
	       "Container grid is %d by %d by %d\n",n,max_id,nx,ny,nz);

	// Import the particles, and create ID lookup tables
	double *xi=new double[max_id+1],*yi=new double[max_id+1],*zi=new double[max_id+1];
	for(j=0;j<n;j++) {
		id=vid[j];x=vx[j];y=vy[j];z=vz[j];
		con.put(id,x,y,z);
		xi[id]=x;
		yi[id]=y;
		zi[id]=z;
	}

	// Open three output files for statistics and gnuplot cells
	fp=safe_fopen("liq-900K.out","w");
	fp2=safe_fopen("liq-900K.gnu","w");
	fp3=safe_fopen("liq-900K-orig.gnu","w");

	// Loop over all particles and compute their Voronoi cells
	voronoicell_neighbor c,c2;
	c_loop_all cl(con);
	if(cl.start()) do if(con.compute_cell(c,cl)) {

		// Get particle position, ID, and neighbor vector
		cl.pos(x,y,z);
		id=cl.pid();
		c.neighbors(neigh);
				
		// Get face areas et total surface of faces
		c.face_areas(vd);c.surface_area();
		c.draw_gnuplot(x,y,z,fp3);
		
		// Initialize second cell
		c2.init(ax-x,bx-x,ay-y,by-y,az-z,bz-z);

		// Add condition on surface: >1% total surface. In addition,
		// skip negative indices, since they correspond to faces
		// against the container boundaries
		for(i=0;i<(signed int) vd.size();i++)
			if(vd[i]>0.01*c.surface_area()&&neigh[i]>=0) {
			j=neigh[i];
			c2.nplane(xi[j]-x,yi[j]-y,zi[j]-z,j);
		}

		// Get information of c2 cell
		c2.face_areas(vd);c2.face_orders(f_order);
		
		// Output information to file
		i=vd.size();
		fprintf(fp,"%d %d",id,i);
		for(j=0;j<i;j++) fprintf(fp," %d",f_order[j]);
		for(j=0;j<i;j++) fprintf(fp," %.3f",vd[j]);		
		fprintf(fp," %.3f %.3f %.3f\n",x,y,z);

		c2.draw_gnuplot(x,y,z,fp2);
	} while (cl.inc());

	// Close files
	fclose(fp);
	fclose(fp2);
	
	// Delete dynamically allocated arrays
	delete [] xi;
	delete [] yi;
	delete [] zi;
}
