#include <cstdio>
#include <cstdlib>
#include <cmath>
using namespace std;

#include "unitcell.hh"
using namespace voro;

int main(int argc,char **argv) {
	unsigned int i,j;
	vector<int> vi;
	vector<double> vd;

	// Check the command line syntax
	if(argc!=7) {
		fprintf(stderr,"Syntax: ./images bx bxy by bxz byz bz\n");
		return VOROPP_CMD_LINE_ERROR;
	}

	// Create the unit cell 
	unitcell uc(atof(argv[1]),atof(argv[2]),atof(argv[3]),atof(argv[4]),atof(argv[5]),atof(argv[6]));

	// Calculate the images
	uc.images(vi,vd);

	// Print the output
	for(i=j=0;i<vd.size();i++,j+=3) printf("%d %d %d %g\n",vi[j],vi[j+1],vi[j+2],vd[i]);
}
