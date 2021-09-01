# Voro++ makefile
#
# Author : Chris H. Rycroft (Harvard University / LBL)
# Email  : chr@alum.mit.edu
# Date   : August 30th 2011

# Load the common configuration file
include ../../config.mk

# List of executables
EXECUTABLES=rad_test finite_sys cylinder_inv single_cell_2d period sphere_mesh lloyd_box import_rahman import_nguyen polycrystal_rahman random_points_10 random_points_200 import_freeman voro_lf split_cell ghost_test neigh_test tri_mesh sphere r_pts_interface minkowski

# Makefile rules
all: $(EXECUTABLES)

%: %.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean
