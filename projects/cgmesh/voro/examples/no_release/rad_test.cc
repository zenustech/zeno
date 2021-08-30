// Voronoi calculation example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

// Set up constants for the container geometry
const double x_min=-1,x_max=1;
const double y_min=-1,y_max=1;
const double z_min=-1,z_max=1;
const double cvol=(x_max-x_min)*(y_max-y_min)*(x_max-x_min);

// Set up the number of blocks that the container is divided into
const int n_x=3,n_y=3,n_z=3;

// Set the number of particles that are going to be randomly introduced
const int particles=1000;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i;
	double pos[particles*4],*posp=pos;
	
	for(i=0;i<particles;i++) {
		*(posp++)=x_min+rnd()*(x_max-x_min);
		*(posp++)=y_min+rnd()*(y_max-y_min);
		*(posp++)=z_min+rnd()*(z_max-z_min);
		*(posp++)=rnd();
	}

	char buf[128];
	for(i=0;i<=200;i++) {
		int j;double x,y,z,r,mul=i*0.01,vol=0;

		container_poly con(x_min,x_max,y_min,y_max,z_min,z_max,n_x,n_y,n_z,
			false,false,false,8);

		posp=pos;
		for(int j=0;j<particles;j++) {
			x=*(posp++);y=*(posp++);z=*(posp++);r=*(posp++)*mul;
			con.put(j,x,y,z,r);
		}

		sprintf(buf,"rad_test_out/fr%d.pov",i);
//		FILE *pf=safe_fopen(buf,"w");
		j=0;
		c_loop_all cl(con);
		voronoicell c;
		cl.start();
		do {
			if(con.compute_cell(c,cl)) {
				vol+=c.volume();
				cl.pos(x,y,z);
//				c.draw_pov(x,y,z,pf);
				j++;
			}
		} while(cl.inc());

		printf("%g %d %g %g\n",mul,j,vol,vol-8);
//		fclose(pf);
	}
}
