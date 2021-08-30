# Voro++ makefile
#
# Author : Chris H. Rycroft (LBL / UC Berkeley)
# Email  : chr@alum.mit.edu
# Date   : May 18th 2011 

# Load the common configuration file
include ../../config.mk

# List of executables
EXECUTABLES=single_cell_2d random_points_2d import_2d doe_diagram

# Makefile rules
all: $(EXECUTABLES) 

single_cell_2d: single_cell_2d.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

random_points_2d: random_points_2d.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

import_2d: import_2d.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

doe_diagram: doe_diagram.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean
