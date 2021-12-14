// Source file for the R2 line class 



// Include files 

#include "R2.h"



// Public variables 

R2Line R2null_line(0.0, 0.0, 0.0);
R2Line R2posx_line(0.0, -1.0, 0.0);
R2Line R2posy_line(1.0, 0.0, 0.0);
R2Line R2negx_line(0.0, 1.0, 0.0);
R2Line R2negy_line(-1.0, 0.0, 0.0);



R2Line::
R2Line(void)
{
}



R2Line::
R2Line(const R2Line& line)
  : vector(line.vector),
    normal(line.normal),
    c(line.c)
{
}



R2Line::
R2Line(double a, double b, double c)
  : vector(-b, a),
    normal(a, b),
    c(c)
{
}



R2Line::
R2Line(double array[3])
  : vector(-array[1], array[0]),
    normal(array[0], array[1]),
    c(array[2])
{
}



R2Line::
R2Line(const R2Point& point, const R2Vector& vector, bool normalized)
{
  // Build line from point and vector
  // Normalize vector if not told that it already is normalized
  this->vector = vector;
  if (!normalized) this->vector.Normalize();
  normal = R2Vector(this->vector.Y(), -(this->vector.X()));
  c = -(normal.X()*point.X() + normal.Y()*point.Y());
}



R2Line::
R2Line(const R2Point& point1, const R2Point& point2)
{
  // Build line through two points
  vector = point2 - point1;
  vector.Normalize();
  normal = R2Vector(this->vector.Y(), -(this->vector.X()));
  c = -(normal.X()*point1.X() + normal.Y()*point1.Y());
}



R2Line::
R2Line(double x1, double y1, double x2, double y2)
{
  // Build line through (x1, y1) and (x2, y2)
  vector.Reset(x2-x1, y2-y1);
  vector.Normalize();
  normal = R2Vector(this->vector.Y(), -(this->vector.X()));
  c = -(normal.X()*x1 + normal.Y()*y1);
}



void R2Line::
Flip(void)
{
  // Flip direction of line
  vector.Flip();
  normal.Flip();
  c = -c;
}



void R2Line::
Mirror(const R2Line& line)
{
  // Mirror line over another line
  R2Point p = (normal * -c).Point();
  p.Mirror(line);
  vector.Mirror(line);
  normal = R2Vector(vector.Y(), -(vector.X()));
  c = -(normal.X()*p.X() + normal.Y()*p.Y());
}



void R2Line::
Translate(const R2Vector& vector)
{
  // Move line by vector
  R2Point point = AnyPoint() + vector;
  c = -(normal.X()*point.X() + normal.Y()*point.Y());
}



void R2Line::
Rotate(const R2Point& origin, double angle)
{
  // Rotate line counter-clockwise around origin by angle
  R2Point point = AnyPoint();
  point.Rotate(origin, angle);
  vector.Rotate(angle);
  Reset(point, vector, true);
}



void R2Line::
Reset(const R2Point& point, const R2Vector& vector, bool normalized)
{
  // Build line from point and vector
  // Normalize vector if not told that it already is normalized
  this->vector = vector;
  if (!normalized) this->vector.Normalize();
  normal = R2Vector(this->vector.Y(), -(this->vector.X()));
  c = -(normal.X()*point.X() + normal.Y()*point.Y());
}


R2Line& R2Line::
operator=(const R2Line& line)
{
  // Copy everything
  vector = line.vector;
  normal = line.normal;
  c = line.c;
  return *this;
}



R2Line R2Line::
operator-(void)
{
  // Return line with flipped orientation
  return R2Line(-A(), -B(), -C());
}



void R2Line::
Print(FILE *fp) const
{
  // Print line parameters
  fprintf(fp, "%g %g %g", A(), B(), C());
}
