# Voro++ makefile
#
# Author : Chris H. Rycroft (LBL / UC Berkeley)
# Email  : chr@alum.mit.edu
# Date   : May 18th 2011 

# Load the common configuration file
include ../../config.mk

# List of executables
EXECUTABLES=nonconvex_cell container_bd comb annulus

# Makefile rules
all: $(EXECUTABLES) 

nonconvex_cell: nonconvex_cell.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

container_bd: container_bd.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

comb: comb.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

annulus: annulus.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean
