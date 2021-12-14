// Source file for the R3 point class 



// Include files 

#include "R3.h"



// Public variables 

const R3Point R3null_point(0.0, 0.0, 0.0);
const R3Point R3ones_point(1.0, 1.0, 1.0);
const R3Point R3posx_point(1.0, 0.0, 0.0);
const R3Point R3posy_point(0.0, 1.0, 0.0);
const R3Point R3posz_point(0.0, 0.0, 1.0);
const R3Point R3negx_point(-1.0, 0.0, 0.0);
const R3Point R3negy_point(0.0, -1.0, 0.0);
const R3Point R3negz_point(0.0, 0.0, -1.0);




// Public functions 

R3Point::
R3Point(void)
{
}



R3Point::
R3Point(double x, double y, double z)
{
  // Initialize coordinates
  v[0] = x; 
  v[1] = y; 
  v[2] = z;
}



R3Point::
R3Point(const R3Point& point)
{
  // Initialize coordinates
  v[0] = point.v[0]; 
  v[1] = point.v[1]; 
  v[2] = point.v[2]; 
}



R3Point::
R3Point(const double array[3])
{
  // Initialize coordinates
  v[0] = array[0]; 
  v[1] = array[1]; 
  v[2] = array[2];
}



R3Vector R3Point::
Vector(void) const
{
  // Return vector to point from origin
  return R3Vector(v[0], v[1], v[2]);
}



void R3Point::
Translate (const R3Vector& vector) 
{
  // Move point by vector
  *this += vector;
}



void R3Point::
Project(const R3Line& line)
{
  // Move point to closest point on line
  const R3Point *p = &(line.Point());
  const R3Vector *v = &(line.Vector());
  double denom = v->Dot(*v);
  if (denom == 0) return;
  double t = (v->X() * (X() - p->X()) + v->Y() *(Y() - p->Y()) + v->Z() * (Z() - p->Z())) / denom;
  *this = *p + *v * t;
}



void R3Point::
Project(const R3Plane& plane)
{
  // Move point to closest point on plane
  double d = R3SignedDistance(plane, *this);
  *this += plane.Normal() * -d;
}



void R3Point::
Mirror(const R3Plane& plane)
{
  // Mirror point across plane
  double d = R3SignedDistance(plane, *this);
  *this += plane.Normal() * (-2.0 * d);
}



void R3Point::
Rotate(const R3Vector& axis, double theta)
{
  // Rotate point counterclockwise around axis through origin by radians ???
  R3Vector v = Vector();
  v.Rotate(axis, theta);
  *this = v.Point();
}



void R3Point::
Rotate(const R3Line& axis, double theta)
{
  // Translate axis to origin
  R3Vector v = *this - axis.Point();

  // Rotate point counterclockwise around axis through origin by radians ???
  v.Rotate(axis.Vector(), theta);

  // Translate axis back from origin
  *this = axis.Point() + v;
}



void R3Point::
Transform(const R3Matrix& matrix)
{
  // Transform point
  *this = matrix * (*this);
}



void R3Point::
InverseTransform(const R3Matrix& matrix)
{
  // Transform point by inverse
  *this = matrix.Inverse() * (*this);
}



R3Point& R3Point::
operator=(const R3Point& point)
{
  // Assign coordinates
  v[0] = point.v[0];
  v[1] = point.v[1];
  v[2] = point.v[2];
  return *this;
}



R3Point& R3Point::
operator+=(const R3Point& point)
{
  // Add coordinates of point 
  v[0] += point[0];
  v[1] += point[1];
  v[2] += point[2];
  return *this;
}



R3Point& R3Point::
operator+=(const R3Vector& vector)
{
  // Add coordinates of vector
  v[0] += vector[0];
  v[1] += vector[1];
  v[2] += vector[2];
  return *this;
}



R3Point& R3Point::
operator-=(const R3Vector& vector)
{
  // Subtract coordinates of vector
  v[0] -= vector[0];
  v[1] -= vector[1];
  v[2] -= vector[2];
  return *this;
}



R3Point& R3Point::
operator*=(const double a)
{
  // Multiply coordinates by scalar 
  v[0] *= a;
  v[1] *= a;
  v[2] *= a;
  return *this;
}



R3Point& R3Point::
operator/=(const double a)
{
  // Divide coordinates by scalar 
  //  assert(!zero(a)); 
  v[0] /= a;
  v[1] /= a;
  v[2] /= a;
  return *this;
}



R3Point 
operator-(const R3Point& point)
{
  // Subtract coordinates of point
  return R3Point(-point.v[0], 
                 -point.v[1], 
                 -point.v[2]);
}



R3Point
operator+(const R3Vector& vector, const R3Point& point)
{
  // Add vector and point
  return point + vector;
}



R3Point 
operator+(const R3Point& point1, const R3Point& point2)
{
  // Add two points
  return R3Point(point1.v[0] + point2.v[0], 
                 point1.v[1] + point2.v[1], 
                 point1.v[2] + point2.v[2]);
}



R3Point 
operator+(const R3Point& point, const R3Vector& vector)
{
  // Add point and vector
  return R3Point(point.X() + vector.X(), 
                 point.Y() + vector.Y(), 
                 point.Z() + vector.Z());
}



R3Vector 
operator-(const R3Point& point1, const R3Point& point2)
{
  // Subtract two points
  return R3Vector(point1.v[0] - point2.v[0], 
                  point1.v[1] - point2.v[1], 
                  point1.v[2] - point2.v[2]);
}



R3Point 
operator-(const R3Point& point, const R3Vector& vector)
{
  // Subtract vector from point
  return R3Point(point.X() - vector.X(), 
                 point.Y() - vector.Y(), 
                 point.Z() - vector.Z());
}



R3Point
operator*(const double a, const R3Point& point)
{
  // Multiply point by scalar
  return point * a;
}



R3Point 
operator*(const R3Point& point, const double a)
{
  // Multiply point by scalar
  return R3Point(point.X() * a, 
                 point.Y() * a,
                 point.Z() * a);
}



R3Point 
operator/(const R3Point& point, const double a)
{
  // Divide point by scalar
  assert(a != 0);
  return R3Point(point.X() / a, 
                 point.Y() / a, 
                 point.Z() / a);
}



void R3Point::
Draw (void) const
{
  // Draw point
}



void R3Point::
Print(FILE *fp) const
{
  // Print point coordinates
  fprintf(fp, "%g %g %g", v[0], v[1], v[2]);
}









