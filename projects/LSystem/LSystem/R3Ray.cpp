// Source file for the R3 ray class 



// Include files 

#include "R3.h"



// Public variables 

const R3Ray R3null_ray(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const R3Ray R3posx_ray(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
const R3Ray R3posy_ray(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
const R3Ray R3posz_ray(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
const R3Ray R3negx_ray(0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
const R3Ray R3negy_ray(0.0, 0.0, 0.0, 0.0, -1.0, 0.0);
const R3Ray R3negz_ray(0.0, 0.0, 0.0, 0.0, 0.0, -1.0);



// Public functions 

R3Ray::
R3Ray(void)
{
}



R3Ray::
R3Ray(const R3Ray& ray)
  : line(ray.line)
{
}



R3Ray::
R3Ray(const R3Point& point, const R3Vector& vector, bool normalized)
  : line(point, vector, normalized)
{
}



R3Ray::
R3Ray(const R3Point& point1, const R3Point& point2)
  : line(point1, point2)
{
}



R3Ray::
R3Ray(double x1, double y1, double z1, double x2, double y2, double z2)
  : line(x1, y1, z1, x2, y2, z2)
{
}



const R3Point R3Ray::
Point(double t) const
{
  // Return point along span
  return (Start() + Vector() * t);
}



const double R3Ray::
T(const R3Point& point) const
{
  // Return parametric value of closest point on ray
  if (IsZero()) return 0.0;
  double denom = Vector().Dot(Vector());
  assert(denom != 0);
  R3Vector topoint = point - Start();
  return (Vector().Dot(topoint) / denom);
}



void R3Ray::
Transform (const R3Matrix& matrix)
{
  // Transform line
  line.Transform(matrix);
}



void R3Ray::
InverseTransform (const R3Matrix& matrix)
{
  // Transform line
  line.InverseTransform(matrix);
}




void R3Ray::
Draw (void) const
{
  // Draw ray
}



void R3Ray::
Print(FILE *fp) const
{
  // Print line containing ray
  line.Print();
}




