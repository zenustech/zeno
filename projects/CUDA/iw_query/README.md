"Find Closest Point on (Tesselated) Surface" Library
====================================================

A Little helper library for performing (reasonably fast) point to
closest point on tessellated surface queries. Input is a "soup" of
triangles, over which we build a simple little spatial-median BVH.
Given this BVH, the user can then query - for one or more points - the
respectively closest surface point for each query point (for each
query point we return the closest point on the surface, the distance
to this point, and the ID of the triangle that contained this point).

Code Overview
- a C99 API (can also be called from fortran) in 'distanceQueries.h'
- makefile will build the implementation of that API into libbvhdq.a
- BVH generation code in bvh.h/bvh.c
- distance query code in distanceQueries.cpp

Dependencies and Prerequisites:
- library dependencies: intentionally none
- compiler: currently requires C++11-support; but should be easy to remove
  this dependency, I only need it for simple thigns like std::atomic
- makefile currently uses intel compiler, but should compile with any other C++11
  compiler


Credits:
- Some of the vector classes were lifted from the OSPRay project; these
  could be stripped down quite a bit (we only need a fraction of that
  functionality), but it was easist to just copy those in and know they'd work...

License:
- Use in whatever way you think fit, without any guarantees or warranties 
  whatsoever. I added a Apache Licence, but any other license would be fine w/ me ...