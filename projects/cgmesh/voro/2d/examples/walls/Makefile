# Voro++ makefile
#
# Author : Chris H. Rycroft (LBL / UC Berkeley)
# Email  : chr@alum.mit.edu
# Date   : May 18th 2011 

# Load the common configuration file
include ../../config.mk

# List of executables
EXECUTABLES=circle polygon

# Makefile rules
all: $(EXECUTABLES) 

circle: circle.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

polygon: polygon.cc
	$(CXX) $(CFLAGS) $(E_INC) $(E_LIB) -o $@ $< -lvoro++_2d

clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean
