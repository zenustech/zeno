// Source file for the R2 point class 



// Include files 

#include "R2.h"



// Public variables 

R2Point R2null_point(0.0, 0.0);
R2Point R2ones_point(1.0, 1.0);
R2Point R2posx_point(1.0, 0.0);
R2Point R2posy_point(0.0, 1.0);
R2Point R2negx_point(-1.0, 0.0);
R2Point R2negy_point(0.0, -1.0);



// Public functions 

R2Point::
R2Point(void)
{
}



R2Point::
R2Point(double x, double y)
{
    v[0] = x; 
    v[1] = y; 
}



R2Point::
R2Point(const R2Point& point)
{
    v[0] = point.v[0]; 
    v[1] = point.v[1]; 
}



R2Point::
R2Point(double array[2])
{
    v[0] = array[0]; 
    v[1] = array[1]; 
}



R2Vector R2Point::
Vector(void) const
{
  // Return vector to point from origin
  return R2Vector(v[0], v[1]);
}



void R2Point::
Project(const R2Line& line)
{
    // Mirror point across line
    double d = R2SignedDistance(*this, line);
    *this += line.Normal() * (-1.0 * d);
}



void R2Point::
Mirror(const R2Line& line)
{
    // Mirror point across line
    double d = R2SignedDistance(*this, line);
    *this += line.Normal() * (-2.0 * d);
}



void R2Point::
Rotate(const R2Point& origin, double angle)
{
    // Rotate point counterclockwise around origin by angle (in radians)
    double x = v[0] - origin[0]; 
    double y = v[1] - origin[1];
    double c = cos(angle);
    double s = sin(angle);
    v[0] = x*c - y*s + origin[0];
    v[1] = x*s + y*c + origin[1];
}



R2Point& R2Point::
operator=(const R2Point& point)
{
    v[0] = point[0];
    v[1] = point[1];
    return *this;
}



R2Point& R2Point::
operator+=(const R2Point& point)
{
    v[0] += point[0];
    v[1] += point[1];
    return *this;
}



R2Point& R2Point::
operator+=(const R2Vector& vector)
{
    v[0] += vector[0];
    v[1] += vector[1];
    return *this;
}



R2Point& R2Point::
operator-=(const R2Vector& vector)
{
    v[0] -= vector[0];
    v[1] -= vector[1];
    return *this;
}



R2Point& R2Point::
operator*=(double a)
{
    v[0] *= a;
    v[1] *= a;
    return *this;
}



R2Point& R2Point::
operator/=(double a)
{
    //  assert(!zero(a)); 
    v[0] /= a;
    v[1] /= a;
    return *this;
}



void R2Point::
Print(FILE *fp) const
{
  // Print point coordinates
  fprintf(fp, "%g %g", v[0], v[1]);
}



R2Point 
operator+(const R2Point& point) 
{
    return point;
}



R2Point 
operator-(const R2Point& point)
{
    return R2Point(-point.X(), 
		   -point.Y());
}



R2Point 
operator+(const R2Point& point1, const R2Point& point2)
{
    return R2Point(point1.v[0] + point2.v[0], 
		   point1.v[1] + point2.v[1]);
}



R2Point 
operator+(const R2Point& point, const R2Vector& vector)
{
    return R2Point(point.X() + vector.X(), 
		   point.Y() + vector.Y());
}



R2Point 
operator+(const R2Vector& vector, const R2Point& point)
{
  // Commute addition
  return (point + vector);
}



R2Vector 
operator-(const R2Point& point1, const R2Point& point2)
{
    return R2Vector(point1.v[0] - point2.v[0], 
		    point1.v[1] - point2.v[1]);
}



R2Point 
operator-(const R2Point& point, const R2Vector& vector)
{
    return R2Point(point.X() - vector.X(), 
		   point.Y() - vector.Y());
}



R2Point 
operator*(const R2Point& point, double a)
{
    return R2Point(point.X() * a, 
		   point.Y() * a);
}



R2Point 
operator/(const R2Point& point, double a)
{
    assert(a != 0);
    return R2Point(point.X() / a, 
		   point.Y() / a);
}



R2Point 
operator*(double a, const R2Point& point)
{
  // Commute scale
  return (point * a);
}



