############################################################################
#
#  Program:         SuperLU
#
#  Module:          make.inc
#
#  Purpose:         Top-level Definitions
#
#  Creation date:   May 10, 2015
#
#  Modified:	    
#		    
#
############################################################################
#
#  The name of the libraries to be created/linked to
#
# SuperLUroot	= /home/bate/zenoblend/build
#
SuperLUroot	= /usr/local
SUPERLULIB   	= $(SuperLUroot)/SRC/libsuperlu.a

TMGLIB       	= libtmglib.a

# BLASDEF 	= -DUSE_VENDOR_BLAS
BLASLIB		= /usr/local/CBLAS/libblas.a
LIBS		= $(SUPERLULIB) /usr/local/CBLAS/libblas.a

#
#  The archiver and the flag(s) to use when building archive (library)
#  If your system has no ranlib, set RANLIB = echo.
#
ARCH         = /usr/bin/ar
ARCHFLAGS    = cr
RANLIB       = /usr/bin/ranlib

CC           = /usr/bin/cc
CFLAGS 	     = -O3 -DNDEBUG -DPRNTlevel=0 -DDEBUGlevel=0  
NOOPTS       = -O0
FORTRAN	     = /usr/bin/gfortran

LOADER       = $(CC)
LOADOPTS     =

#
# The directory in which Matlab is installed
#
MATLAB	     = /Applications/MATLAB_R2015b.app
