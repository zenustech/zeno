// Include files for R3 package
#ifndef R3_INCLUDED
#define R3_INCLUDED



// Include files 

#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <cassert>
#include <cstring>
#include <cctype>
#include <cmath>
#include <climits>
#include <algorithm>
using namespace std;



// Constant declarations

#define R3_X 0
#define R3_Y 1
#define R3_Z 2



// Class declarations

class R3Point;
class R3Vector;
class R3Line;
class R3Ray;
class R3Segment;
class R3Plane;
class R3Box;
class R3Sphere;
class R3Matrix;

#include "R2.h"

// Class include files
#include "R3Point.h"
#include "R3Vector.h"
#include "R3Line.h"
#include "R3Ray.h"
#include "R3Segment.h"
#include "R3Plane.h"
#include "R3Box.h"
#include "R3Matrix.h"



// Utility include files

#include "R3Distance.h"



#endif
