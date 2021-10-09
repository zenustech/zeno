// Voro++, a 3D cell-based Voronoi library
//
// Author   : Chris H. Rycroft (LBL / UC Berkeley)
// Email    : chr@alum.mit.edu
// Date     : July 1st 2008

#include <cstdlib>

const int n_table=2;

const char rad_ctable[][4]={
	"O",
	"Si"
};

const double rad_table[]={
	0.8,
	0.5
};

double radial_lookup(char *buffer) {
	for(int i=0;i<n_table;i++) if(strcmp(rad_ctable[i],buffer)==0) return rad_table[i];
	fprintf(stderr,"Entry \"%s\" not found in table\n",buffer);
	exit(VOROPP_FILE_ERROR);
}
