
EL TOPO: ROBUST TOPOLOGICAL OPERATIONS FOR DYNAMIC EXPLICIT SURFACES
===============================================================================

El Topo is a C-callable library built in C++.  This readme describes:

- how to build the library
- how to use the library

(Also included in this release is code for creating an executable with various 
example applications.  See the talpa directory for more details.)


Building the library:
=====================

1) Create Makefile.local_defs

Makefile should handle building El Topo library on Linux and OS/X with g++.  It 
reads a file called Makefile.local_defs, which contains platform-specific 
definitions.  You must create this file!  Makefile.example_defs includes 
suggested settings for Linux and OS/X.

2) Generate dependencies and compile

Once you have created a file called Makefile.local_defs, you can build the El 
Topo library by running "make depend", followed by "make release" or 
"make debug".  This will create a file called "libeltopo_release.a" or 
"libeltopo_debug.a" which can be used by your C/C++ program.

Example:
$> make depend
$> make release

3) Link to your program

The library is written in C++, so if you are using it from a C program, you must
link against the standard C++ library when building your program.  It also 
requires the BLAS and LAPACK libraries.

Example:
$> gcc main.c libeltopo_release.a -llapack -lblas -lstdc++ -o your_executable


Using the library:
=====================

The files eltopo.h and eltopo.cpp define the interface for El Topo.  There are 
two main functions,  el_topo_static_operations() and el_topo_integrate(), which 
take as parameters input and output vertex coordinates, triangles, and masses, 
as well as some special structures.  These structures define the parameters to 
be used by El Topo (see eltopo.h for more details).  The function 
el_topo_static_operations() performs mesh improvement and topological changes 
such as merging and separation, and el_topo_integrate() moves the surface 
vertices while guaranteeing no self-intersections.


A tour of the code base:
=====================

Our method is outlined in our SISC paper [Brochu and Bridson 2009], however it 
may be helpful to browse our code base.  The main classes you probably want to 
look at are:

NonDestructiveTriMesh
DynamicSurface
SurfTrack

NonDestructiveTriMesh
---------------------

This is a basic triangle mesh class.  The fundamental data is simply a list of 
triangles.  It is "nondestructive" in that when you remove a triangle, it marks 
the triangle as deleted, but doesn't change the size of the list.  The list of 
triangles can then be defragmented as necessary.

There is a set of auxiliary data structures containing various incidence 
relations.  For example, vtxtri contains, for each vertex, the set of triangles 
incident on that vertex.  These structures are useful for getting around the 
mesh, but must be updated when the set of triangles changes.

We generally defrag the list of triangles once per frame if the connectivity 
changes, then rebuild the auxiliary data structures.

DynamicSurface
---------------------

Main data members of this class are the mesh (NonDestructiveTriMesh) and a set 
of vertex locations.  Additional data members include per-vertex data such as 
velocities and masses.

Most important member functions are collision detection and resolution 
functions.  This class contains enough functionality to advect a surface from 
one time step to the next in an intersection-free state, without changing 
topology or connectivity.

(This class would be sufficient for representing cloth if no mesh refinement 
was required.)

SurfTrack
---------------------

A child class of DynamicSurface.  This class contains functions for mesh 
adaptivity and topological changes.

Other classes and files:
=====================

BroadPhase, BroadPhaseGrid and AccelerationGrid
---------------------

The acceleration structure for broad phase collision detection.  It currently 
consists of three regular grids, one grid each for triangles, edges and 
vertices.  Other broad phase approaches could be implemented by subclassing the 
BroadPhase base class.

SubdivisionScheme
---------------------

An interface for interpolating subdivision schemes.  We currently use 
ButterflySubdivision for all our examples.

Common:
---------------------

Common is a set of files shared by our research group.  It contains several 
useful classes and functions, notably:

vec: A templated n-dimensional vector class.  For example, Vec3d (a vector of 
3 doubles) is used all over the place.
mat: A templated matrix class.
gluvi: An OpenGL GUI.
blas_wrapper and lapack_wrapper: Cross-platform interfaces to BLAS and LAPACK 
functions.


References
=====================

[Brochu and Bridson 2009]: Tyson Brochu and Robert Bridson, Robust Topological 
Operations for Dynamic Explicit Surfaces, SIAM J. Sci. Comput., vol. 31, no. 4 
(2009), pp. 2472-2493 

