// Source file for the R3 Box class 



// Include files 

#include "R3.h"



// Public variables 

const R3Box R3zero_box(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
const R3Box R3unit_box(-1.0, -1.0, -1.0, 1.0, 1.0, 1.0);
const R3Box R3null_box( INT_MAX, INT_MAX, INT_MAX, 
		       -INT_MAX, -INT_MAX, -INT_MAX);



// Public functions 

R3Box::
R3Box(void)
{
}



R3Box::
R3Box(const R3Box& box)
  : minpt(box.minpt),
    maxpt(box.maxpt)
{
}



R3Box::
R3Box(const R3Point& minpt, const R3Point& maxpt)
  : minpt(minpt), 
    maxpt(maxpt)
{
}



R3Box::
R3Box(double xmin, double ymin, double zmin,
      double xmax, double ymax, double zmax)
  : minpt(xmin, ymin, zmin),
    maxpt(xmax, ymax, zmax)
{
}



R3Point R3Box::
Corner (int xdir, int ydir, int zdir) const
{
  // Return corner point 
  return R3Point(Coord(xdir, R3_X), Coord(ydir, R3_Y), Coord(zdir, R3_Z));
}



int R3Box::
ShortestAxis (void) const
{
  // Compute length of each axis
  double dx = this->XLength();
  double dy = this->YLength();
  double dz = this->ZLength();

  // Return shortest axis
  if (dx < dy) {
    if (dx < dz) return R3_X;
    else return R3_Z;
  }
  else {
    if (dy < dz) return R3_Y;
    else return R3_Z;
  }    
}



int R3Box::
LongestAxis (void) const
{
  // Compute length of each axis
  double dx = this->XLength();
  double dy = this->YLength();
  double dz = this->ZLength();

  // Return longest axis
  if (dx > dy) {
    if (dx > dz) return R3_X;
    else return R3_Z;
  }
  else {
    if (dy > dz) return R3_Y;
    else return R3_Z;
  }    
}



double R3Box::
ShortestAxisLength (void) const
{
  // Compute length of each axis
  double dx = this->XLength();
  double dy = this->YLength();
  double dz = this->ZLength();

  // Return length of shortest axis
  if (dx < dy) {
    if (dx < dz) return dx;
    else return dz;
  }
  else {
    if (dy < dz) return dy;
    else return dz;
  }    
}



double R3Box::
LongestAxisLength (void) const
{
  // Compute length of each axis
  double dx = this->XLength();
  double dy = this->YLength();
  double dz = this->ZLength();

  // Return longest axis
  if (dx > dy) {
    if (dx > dz) return dx;
    else return dz;
  }
  else {
    if (dy > dz) return dy;
    else return dz;
  }    
}



double R3Box::
DiagonalLength (void) const
{
  // Return length of Box along diagonal
  return R3Distance(minpt, maxpt);
}



R3Point R3Box::
ClosestPoint(const R3Point& point) const
{
  // Return closest point in box
  R3Point closest(point);
  if (closest.X() < XMin()) closest[R3_X] = XMin();
  else if (closest.X() > XMax()) closest[R3_X] = XMax();
  if (closest.Y() < YMin()) closest[R3_Y] = YMin();
  else if (closest.Y() > YMax()) closest[R3_Y] = YMax();
  if (closest.Z() < ZMin()) closest[R3_Z] = ZMin();
  else if (closest.Z() > ZMax()) closest[R3_Z] = ZMax();
  return closest;
}



R3Point R3Box::
Centroid (void) const
{
  // Return center point
  return R3Point(XCenter(), YCenter(), ZCenter());
}



void R3Box::
Empty (void) 
{
  // Copy empty box 
  *this = R3null_box;
}



void R3Box::
Translate(const R3Vector& vector)
{
  // Move box by vector
  minpt += vector;
  maxpt += vector;
}



void R3Box::
Union (const R3Point& point) 
{
  // Expand this to include point
  if (minpt.X() > point.X()) minpt[0] = point.X();
  if (minpt.Y() > point.Y()) minpt[1] = point.Y();
  if (minpt.Z() > point.Z()) minpt[2] = point.Z();
  if (maxpt.X() < point.X()) maxpt[0] = point.X();
  if (maxpt.Y() < point.Y()) maxpt[1] = point.Y();
  if (maxpt.Z() < point.Z()) maxpt[2] = point.Z();
}



void R3Box::
Union (const R3Box& box) 
{
  // Expand this to include box
  if (minpt.X() > box.XMin()) minpt[0] = box.XMin();
  if (minpt.Y() > box.YMin()) minpt[1] = box.YMin();
  if (minpt.Z() > box.ZMin()) minpt[2] = box.ZMin();
  if (maxpt.X() < box.XMax()) maxpt[0] = box.XMax();
  if (maxpt.Y() < box.YMax()) maxpt[1] = box.YMax();
  if (maxpt.Z() < box.ZMax()) maxpt[2] = box.ZMax();
}



void R3Box::
Intersect (const R3Box& box) 
{
  // Intersect with box
  if (minpt.X() < box.XMin()) minpt[0] = box.XMin();
  if (minpt.Y() < box.YMin()) minpt[1] = box.YMin();
  if (minpt.Z() < box.ZMin()) minpt[2] = box.ZMin();
  if (maxpt.X() > box.XMax()) maxpt[0] = box.XMax();
  if (maxpt.Y() > box.YMax()) maxpt[1] = box.YMax();
  if (maxpt.Z() > box.ZMax()) maxpt[2] = box.ZMax();
}



void R3Box::
Transform (const R3Matrix& matrix)
{
  // Do not transform empty box
  if (IsEmpty()) return;

  // Transform box 
  R3Box tmp = R3null_box;
  tmp.Union(matrix * Corner(0,0,0));
  tmp.Union(matrix * Corner(0,0,1));
  tmp.Union(matrix * Corner(0,1,0));
  tmp.Union(matrix * Corner(0,1,1));
  tmp.Union(matrix * Corner(1,0,0));
  tmp.Union(matrix * Corner(1,0,1));
  tmp.Union(matrix * Corner(1,1,0));
  tmp.Union(matrix * Corner(1,1,1));
  *this = tmp;
}



void R3Box::
Reset(const R3Point& min, const R3Point& max)
{
  // Move box by vector
  minpt = min;
  maxpt = max;
}




void R3Box::
Draw (void) const
{
  // Draw box faces
}



void R3Box::
Outline (void) const
{
  // Draw box edges
}



void R3Box::
Print(FILE *fp) const
{
  // Print min and max points
  fprintf(fp, "(%g %g %g) (%g %g %g)", minpt[0], minpt[1], minpt[2], maxpt[0], maxpt[1], maxpt[2]);
}




