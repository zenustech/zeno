// Source file for the R3 plane class 



// Include files 

#include "R3.h"



// Public variables 

const R3Plane R3null_plane(0.0, 0.0, 0.0, 0.0);
const R3Plane R3posyz_plane(1.0, 0.0, 0.0, 0.0);
const R3Plane R3posxz_plane(0.0, 1.0, 0.0, 0.0);
const R3Plane R3posxy_plane(0.0, 0.0, 1.0, 0.0);
const R3Plane R3negyz_plane(-1.0, 0.0, 0.0, 0.0);
const R3Plane R3negxz_plane(0.0, -1.0, 0.0, 0.0);
const R3Plane R3negxy_plane(0.0, 0.0, -1.0, 0.0);



// Public functions 

R3Plane::
R3Plane(void)
{
}



R3Plane::
R3Plane(const R3Plane& plane)
  : v(plane.v), 
    d(plane.d)
{
}



R3Plane::
R3Plane(double a, double b, double c, double d)
  : v(a, b, c), 
    d(d)
{
  v.Normalize();
}



R3Plane::
R3Plane(const double a[4])
  : v(&a[0]), 
    d(a[3])
{
  v.Normalize();
}



R3Plane::
R3Plane(const R3Vector& normal, double d)
  : v(normal),
    d(d)
{
  v.Normalize();
}



R3Plane::
R3Plane(const R3Point& point, const R3Vector& normal)
{
  // Construct plane from point and normal vector
  v = normal;
  v.Normalize();
  d = -(v[0]*point[0] + v[1]*point[1] + v[2]*point[2]);
}



R3Plane::
R3Plane(const R3Point& point, const R3Line& line)
{
  // Construct plane through point and line
  v = point - line.Point();
  v.Cross(line.Vector());
  v.Normalize();
  d = -(v[0]*point[0] + v[1]*point[1] + v[2]*point[2]);
}



R3Plane::
R3Plane(const R3Point& point, const R3Vector& vector1, const R3Vector& vector2)
{
  // Construct plane through point and two vectors
  v = vector1 % vector2;
  v.Normalize();
  d = -(v[0]*point[0] + v[1]*point[1] + v[2]*point[2]);
}



R3Plane::
R3Plane(const R3Point& point1, const R3Point& point2, const R3Point& point3)
{
  // Construct plane through three points
  v = point2 - point1;
  R3Vector v3 = point3 - point1;
  v.Cross(v3);
  v.Normalize();
  d = -(v[0]*point1[0] + v[1]*point1[1] + v[2]*point1[2]);
}



R3Plane::
R3Plane(const R3Point *points, int npoints)
{
  // Check number of points
  if (npoints < 3) {
    v = R3zero_vector;
    d = 0;
  }
  else {
    // Compute best normal for counter-clockwise array of points using newell's method 
    v = R3null_vector;
    R3Point c = R3null_point;
    const R3Point *p1 = &points[npoints-1];
    for (int i = 0; i < npoints; i++) {
      const R3Point *p2 = &points[i];
      v[0] += (p1->Y() - p2->Y()) * (p1->Z() + p2->Z());
      v[1] += (p1->Z() - p2->Z()) * (p1->X() + p2->X());
      v[2] += (p1->X() - p2->X()) * (p1->Y() + p2->Y());
      c+= *p2;
      p1 = p2;
    }

    // Normalize 
    v.Normalize();
    c /= npoints;

    // Compute d from centroid and normal
    d = -(v[0]*c[0] + v[1]*c[1] + v[2]*c[2]);
  }
}



R3Point R3Plane::
Point (void) const
{
  // Return point on plane
  return R3zero_point + v * -d;
}



void R3Plane::
Mirror(const R3Plane& plane)
{
  // Mirror plane ???
  R3Point p = Point();
  p.Mirror(plane);
  v.Mirror(plane);
  Reposition(p);
}



void R3Plane::
Reposition(const R3Point& point)
{
  // Move plane
  d = -(v[0]*point[0] + v[1]*point[1] + v[2]*point[2]);
}



void R3Plane::
Translate(const R3Vector& vector) 
{
  // Move plane by vector - there's got to be a better way ???
  Reposition(Point() + vector);
}



void R3Plane::
Transform (const R3Matrix& matrix)
{
  // Transform plane
  R3Point p = Point();
  R3Matrix m = matrix.Inverse().Transpose();
  p.Transform(matrix);
  v.Transform(m);
  Reposition(p);
}



void R3Plane::
InverseTransform (const R3Matrix& matrix)
{
  // Transform plane by inverse
  R3Point p = Point();
  R3Matrix m = matrix.Transpose();
  p.Transform(matrix);
  v.Transform(m);
  Reposition(p);
}



void R3Plane::
Reset(const R3Point& point, const R3Vector& normal) 
{
  // Reset plane
  v = normal;
  v.Normalize();
  d = -(v[0]*point[0] + v[1]*point[1] + v[2]*point[2]);
}





