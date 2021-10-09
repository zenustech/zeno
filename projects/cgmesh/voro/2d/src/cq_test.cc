#include <cstdio>
#include <cmath>

#include "voro++_2d.hh"
using namespace voro;

#include "omp.h"

// This function returns a random floating point number between 0 and 1
double rnd() {return double(rand())/RAND_MAX;}

int main() {
	int i,l,n=10;double x,y,r,t1,t2;

	while(n<10000000) {
		container_quad_2d con1(-1,1,-1,1);
		l=int(sqrt(double(n))/3.46)+1;
		container_2d con2(-1,1,-1,1,l,l,false,false,8);

		for(i=0;i<n;i++) {
			x=2*rnd()-1;
			y=2*rnd()-1;
			r=1;//(x*x+y*y)*0.5;
			con1.put(i,x*r,y*r);
			con2.put(i,x*r,y*r);
		}

		con1.setup_neighbors();
		t2=omp_get_wtime();
		con2.compute_all_cells();
		t2=omp_get_wtime()-t2;
		t1=omp_get_wtime();
		con1.compute_all_cells();
		t1=omp_get_wtime()-t1;

		printf("%d %g %g\n",n,t1,t2);
		n+=n>>2;
	}
}
