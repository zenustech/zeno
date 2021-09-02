#include <cmath>
using namespace std;

#include "voro++_2d.hh"
using namespace voro;

const double pi=3.1415926535897932384626433832795;
const double radius=0.5;
const int n=5;

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i=0;double x,y,arg;

	// Initialize the container class to be the unit square, with
	// non-periodic boundary conditions. Divide it into a 10 by 10 grid, with
	// an initial memory allocation of 8 particles per grid square.
	container_2d con(-1,1,-1,1,10,10,false,false,8);

	// Add circular wall object
	wall_plane_2d *wc[n];
	for(i=0,arg=0;i<n;i++,arg+=2*pi/n) {
		wc[i]=new wall_plane_2d(sin(arg),-cos(arg),radius);
		con.add_wall(wc[i]);
	}

	// Add 1000 random points to the container
	while(i<1000) {
		x=2*rnd()-1;
		y=2*rnd()-1;
		if(con.point_inside(x,y)) {
			con.put(i,x,y);
			i++;
		}
	}

	// Output the particle positions to a file
	con.draw_particles("polygon.par");

	// Output the Voronoi cells to a file, in the gnuplot format
	con.draw_cells_gnuplot("polygon.gnu");

	// Sum the Voronoi cell areas and compare to the circle area
	double carea=n*radius*radius*tan(pi/n),varea=con.sum_cell_areas();
	printf("Total polygon area      : %g\n"
	       "Total Voronoi cell area : %g\n"
	       "Difference              : %g\n",carea,varea,varea-carea);

	// Since they were dynamically allocated, delete the wall_plane_2d
	// objects
	for(i=0;i<n;i++) delete wc[i];
}
