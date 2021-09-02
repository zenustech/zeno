include ../make.inc

#######################################################################
#  This makefile creates the Fortran example interface to use the
#  C routines in SuperLU.
#######################################################################

HEADER   = ../SRC
LIBS	+= -lm

# double real
DF77EXM	= f77_main.o hbcode1.o c_fortran_dgssv.o
TEST_OMP= test_omp.o hbcode1.o c_fortran_dgssv.o

# double complex
ZF77EXM	= z_f77_main.o zhbcode1.o c_fortran_zgssv.o

all:	df77exm zf77exm test_omp

df77exm: $(DF77EXM) $(SUPERLULIB)
	$(FORTRAN) $(LOADOPTS) $(DF77EXM) $(LIBS) -o $@

zf77exm: $(ZF77EXM) $(SUPERLULIB)
	$(FORTRAN) $(LOADOPTS) $(ZF77EXM) $(LIBS) -o $@

test_omp: $(TEST_OMP) $(SUPERLULIB)
	$(FORTRAN) $(LOADOPTS) $(TEST_OMP)  $(LIBS) -o $@

.c.o:
	$(CC) $(CFLAGS) $(CDEFS) -I$(HEADER) -c $< $(VERBOSE)

.f.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

.F.o:
	$(FORTRAN) $(FFLAGS) -c $< $(VERBOSE)

clean:	
	rm -f *.o *f77exm test_omp

