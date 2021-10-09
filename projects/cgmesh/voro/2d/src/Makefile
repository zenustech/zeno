# Voro++ makefile
#
# Author : Chris H. Rycroft (Harvard SEAS / LBL)
# Email  : chr@alum.mit.edu
# Date   : February 27th 2008

# Load the common configuration file
include ../config.mk

# List of the common source files
objs=common.o cell_2d.o container_2d.o v_base_2d.o v_compute_2d.o \
     c_loops_2d.o wall_2d.o cell_nc_2d.o ctr_boundary_2d.o ctr_quad_2d.o \
     quad_march.o
src=$(patsubst %.o,%.cc,$(objs))
execs=cq_test

# Makefile rules
all: libvoro++_2d.a $(execs)

depend:
	$(CXX) -MM $(src) >Makefile.dep

include Makefile.dep

worklist_2d.hh v_base_wl_2d.cc: worklist_gen.pl
	perl worklist_gen.pl

libvoro++_2d.a: $(objs)
	rm -f libvoro++_2d.a
	ar rs libvoro++_2d.a $^

cq_test: cq_test.cc libvoro++_2d.a
	$(CXX) -fopenmp $(CFLAGS) -o $@ $^

%.o: %.cc
	$(CXX) $(CFLAGS) -c $<

help: Doxyfile $(SOURCE)
	doxygen Doxyfile
	cd ../latex && $(MAKE)

clean:
	rm -f $(objs) libvoro++_2d.a

.PHONY: all help depend
