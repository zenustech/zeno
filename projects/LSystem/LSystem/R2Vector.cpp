// Source file for the R2 vector class 



// Include files 

#include "R2.h"



// Public variables 

R2Vector R2null_vector(0.0, 0.0);
R2Vector R2ones_vector(1.0, 1.0);
R2Vector R2posx_vector(1.0, 0.0);
R2Vector R2posy_vector(0.0, 1.0);
R2Vector R2negx_vector(-1.0, 0.0);
R2Vector R2negy_vector(0.0, -1.0);



// Public functions 

R2Vector::
R2Vector(void)
{
}



R2Vector::
R2Vector(double x, double y)
{
  v[0] = x; 
  v[1] = y; 
}



R2Vector::
R2Vector(const R2Vector& vector)
{
  v[0] = vector.v[0]; 
  v[1] = vector.v[1]; 
}



R2Vector::
R2Vector(double array[2])
{
  v[0] = array[0]; 
  v[1] = array[1]; 
}



void R2Vector::
Normalize(void)
{
  double length = Length();
  if (length == 0.0) return;
  v[0] /= length;
  v[1] /= length;
}



void R2Vector::
Flip (void) 
{
  // Flip vector direction
  v[0] = -v[0];
  v[1] = -v[1];
}



void R2Vector::
Scale(double a)
{
  v[0] *= a;
  v[1] *= a;
}



void R2Vector::
Rotate(double angle)
{
  // Rotate vector counterclockwise 
  double c = cos(angle);
  double s = sin(angle);
  double x = v[0], y = v[1];
  v[0] = c*x - s*y;
  v[1] = s*x + c*y;
}



void R2Vector::
Project(const R2Vector& vector)
{
  // Project onto another vector    
  double dot = Dot(vector);
  double length = vector.Length();
  if (length != 0) dot /= (length * length);
  *this = vector * dot;
}



void R2Vector::
Mirror(const R2Line& line)
{
  // Mirror vector across line
  double d = Dot(line.Normal());
  *this += line.Normal() * (-2.0 * d);
}



R2Vector& R2Vector::
operator=(const R2Vector& vector)
{
  v[0] = vector.v[0];
  v[1] = vector.v[1];
  return *this;
}



R2Vector& R2Vector::
operator+=(const R2Vector& vector)
{
  v[0] += vector.v[0];
  v[1] += vector.v[1];
  return *this;
}



R2Vector& R2Vector::
operator-=(const R2Vector& vector)
{
  v[0] -= vector.v[0];
  v[1] -= vector.v[1];
  return *this;
}



R2Vector& R2Vector::
operator*=(double a)
{
  v[0] *= a;
  v[1] *= a;
  return *this;
}



R2Vector& R2Vector::
operator*=(const R2Vector& vector)
{
  // Entry by entry multiply (not dot or cross product)
  v[0] *= vector.v[0];
  v[1] *= vector.v[1];
  return *this;
}



R2Vector& R2Vector::
operator/=(double a)
{
  assert(a != 0);
  v[0] /= a;
  v[1] /= a;
  return *this;
}



R2Vector& R2Vector::
operator/=(const R2Vector& vector)
{
  // Entry by entry divide
  assert(vector.v[0] != 0);
  assert(vector.v[1] != 0);
  v[0] /= vector.v[0];
  v[1] /= vector.v[1];
  return *this;
}



void R2Vector::
Print(FILE *fp) const
{
  // Print vector coordinates
  fprintf(fp, "%g %g", v[0], v[1]);
}



R2Vector 
operator+(const R2Vector& vector) 
{
  return vector;
}



R2Vector 
operator-(const R2Vector& vector)
{
  return R2Vector(-vector.X(), 
                  -vector.Y());
}



R2Vector 
operator+(const R2Vector& vector1, const R2Vector& vector2)
{
  return R2Vector(vector1.v[0] + vector2.v[0], 
                  vector1.v[1] + vector2.v[1]);
}



R2Vector 
operator-(const R2Vector& vector1, const R2Vector& vector2)
{
  return R2Vector(vector1.v[0] - vector2.v[0], 
                  vector1.v[1] - vector2.v[1]);
}



R2Vector 
operator*(const R2Vector& vector1, const R2Vector& vector2)
{
  // Entry by entry multiply (not dot or cross product)
  return R2Vector(vector1.v[0] * vector2.v[0], 
                  vector1.v[1] * vector2.v[1]);
}



R2Vector 
operator*(const R2Vector& vector, double a)
{
  return R2Vector(vector.X() * a, 
                  vector.Y() * a);
}



R2Vector 
operator/(const R2Vector& vector1, const R2Vector& vector2)
{
  assert(vector2.v[0] != 0);
  assert(vector2.v[1] != 0);
  return R2Vector(vector1.v[0] / vector2.v[0], 
                  vector1.v[1] / vector2.v[1]);
}



R2Vector 
operator/(const R2Vector& vector, double a)
{
  assert(a != 0);
  return R2Vector(vector.X() / a, 
                  vector.Y() / a);
}



double
operator%(const R2Vector& vector1, const R2Vector& vector2)
{
  // Return cross product
  return vector1.X()*vector2.Y() - vector1.Y()*vector2.X();
}



