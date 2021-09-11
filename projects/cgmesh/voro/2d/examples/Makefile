# Voro++ makefile
#
# Author : Chris H. Rycroft (LBL / UC Berkeley)
# Email  : chr@alum.mit.edu
# Date   : August 30th 2011

# Makefile rules
all: ex_basic ex_walls ex_boundary ex_extra

ex_basic:
	$(MAKE) -C basic

ex_walls:
	$(MAKE) -C walls

ex_boundary:
	$(MAKE) -C boundary

ex_extra:
	$(MAKE) -C extra

clean:
	$(MAKE) -C basic clean
	$(MAKE) -C walls clean
	$(MAKE) -C boundary clean
	$(MAKE) -C extra clean

.PHONY: all ex_basic ex_walls ex_boundary ex_extra clean
