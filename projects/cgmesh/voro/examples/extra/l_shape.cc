// Irregular packing example code
//
// Author   : Chris H. Rycroft (Harvard University / LBL)
// Email    : chr@alum.mit.edu
// Date     : August 30th 2011

#include "voro++.hh"
using namespace voro;

// Set the number of particles that are going to be randomly introduced
const int particles=20;

// This function returns a random double between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

// Create a wall class that will initialize the Voronoi cell to fill the
// L-shaped domain
class wall_l_shape : public wall {
	public:
		wall_l_shape() {
			v.init_l_shape();
			v.draw_gnuplot(0,0,0,"l_shape_init.gnu");
		};
		bool point_inside(double x,double y,double z) {return true;}
		bool cut_cell(voronoicell &c,double x,double y,double z) {

			// Set the cell to be equal to the L-shape
			c=v;
			c.translate(-x,-y,-z);

			// Set the tolerance to 100, to make the code search
			// for cases where non-convex cells are cut in multiple
			// places
			c.big_tol=100;
			return true;
		}
		bool cut_cell(voronoicell_neighbor &c,double x,double y,double z) {

			// Set the cell to be equal to the L-shape
			c=v;
			c.translate(-x,-y,-z);
			
			// Set the tolerance to 100, to make the code search
			// for cases where non-convex cells are cut in multiple
			// places
			c.big_tol=100;
			return true;
		}
	private:
		voronoicell v;
};

int main() {
	int i=0;
	double x,y,z;

	// Create a container
	container con(-1,1,-1,1,-1,1,5,5,5,false,false,false,8);

	// Create the L-shape wall class and add it to the container
	wall_l_shape(wls);
	con.add_wall(wls);

	// Add particles, making sure not to place any outside of the L-shape 
	while(i<particles) {
		x=2*rnd()-1;
		y=2*rnd()-1;
		if(x<0&&y>0) continue;
		z=2*rnd()-1;
		con.put(i,x,y,z);
		i++;
	}

	// Check the Voronoi cell volume; it should be 6
	printf("Voronoi cell volume: %.8g\n",con.sum_cell_volumes());

	// Save the particles and Voronoi cells in gnuplot format
	con.draw_particles("l_shape_p.gnu");
	con.draw_cells_gnuplot("l_shape_v.gnu");
}
