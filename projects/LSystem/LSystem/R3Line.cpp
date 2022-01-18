// Source file for the R3 line class 



// Include files 

#include "R3.h"



// Public variables 

const R3Line R3null_line(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const R3Line R3posx_line(0.0, 0.0, 0.0, 1.0, 0.0, 0.0);
const R3Line R3posy_line(0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
const R3Line R3posz_line(0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
const R3Line R3negx_line(0.0, 0.0, 0.0, -1.0, 0.0, 0.0);
const R3Line R3negy_line(0.0, 0.0, 0.0, 0.0, -1.0, 0.0);
const R3Line R3negz_line(0.0, 0.0, 0.0, 0.0, 0.0, -1.0);



// Public functions 

R3Line::
R3Line(void)
{
}



R3Line::
R3Line(const R3Line& line)
  : point(line.point),
    vector(line.vector)
{
}



R3Line::
R3Line(const R3Point& point, const R3Vector& vector, bool normalized)
  : point(point),
    vector(vector)
{
  // Normalize vector
  if (!normalized) this->vector.Normalize();
}



R3Line::
R3Line(const R3Point& point1, const R3Point& point2)
  : point(point1),
    vector(point2 - point1)
{
  // Normalize vector
  vector.Normalize();
}



R3Line::
R3Line(double x1, double y1, double z1, double x2, double y2, double z2)
  : point(x1, y1, z1),
    vector(x2-x1, y2-y1, z2-z1)
{
  // Normalize vector
  vector.Normalize();
}




void R3Line::
Transform (const R3Matrix& matrix)
{
  // Transform point and vector
  point = matrix * point;
  vector = matrix * vector;
  vector.Normalize();
}



void R3Line::
InverseTransform (const R3Matrix& matrix)
{
  // Transform point and vector
  R3Matrix inverse = matrix.Inverse();
  Transform(inverse);
}



bool R3Line::
operator==(const R3Line& line) const
{
  // Check if vectors are equal
  if (vector != line.vector) return false;

  // Return whether point lies on this line
  R3Vector v = vector;
  v.Cross(line.Point() - point);
  return v.IsZero();
}




void R3Line::
Draw (void) const
{
  // Draw line
}



void R3Line::
Print(FILE *fp) const
{
  // Print point and vector
  fprintf(fp, "(%g %g %g) (%g %g %g)", point[0], point[1], point[2], vector[0], vector[1], vector[2]);
}




