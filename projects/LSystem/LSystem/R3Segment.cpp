// Source file for the R3 segment class 



// Include files 

#include "R3.h"



// Public variables 

R3Segment R3null_segment(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);



// Public functions 

R3Segment::
R3Segment(void)
{
}



R3Segment::
R3Segment(const R3Segment& segment)
  : ray(segment.ray),
    end(segment.end),
    length(segment.length)
{
}



R3Segment::
R3Segment(const R3Point& point, const R3Vector& vector)
  : end(point + vector)
{
  R3Vector v(vector);
  length = v.Length();
  if (length != 0) v /= length;
  ray.Reset(point, v, true);
}



R3Segment::
R3Segment(const R3Point& point1, const R3Point& point2)
  : end(point2)
{
  R3Vector v(point2 - point1);
  length = v.Length();
  if (length != 0) v /= length;
  ray.Reset(point1, v, true);
}



R3Segment::
R3Segment(double x1, double y1, double z1, double x2, double y2, double z2)
  : end(x2, y2, z2)
{
  R3Point p1(x1, y1, z1);
  R3Point p2(x2, y2, z2);
  R3Vector v(p2 - p1);
  length = v.Length();
  if (length != 0) v /= length;
  ray.Reset(p1, v, true);
}



const R3Point R3Segment::
Point(double t) const
{
  // Return point along segment
  return (Start() + Vector() * t);
}



const R3Point R3Segment::
Midpoint(void) const
{
  // Return midpoint of segment
  return (Start() + End()) * 0.5;
}



const R3Box R3Segment::
BBox(void) const
{
  // Return bounding box 
  R3Box bbox(Start(), Start());
  bbox.Union(End());
  return bbox;
}



const double R3Segment::
T(const R3Point& point) const
{
  // Return parametric value of closest point on segment
  if (length == 0.0) return 0.0;
  R3Vector topoint = point - Start();
  return Vector().Dot(topoint);
}



const bool R3Segment::
IsPoint(void) const
{
  // Return whether segment covers a single point
  return (length == 0.0);
}



void R3Segment::
Reposition(int k, const R3Point& point)
{
  // Set one endpoint of segment
  if (k == 0) ray = R3Ray(point, end);
  else { end = point; ray.Align(end - ray.Start()); }
  length = R3Distance(Start(), End());
}



void R3Segment::
Align(const R3Vector& vector)
{
  // Set vector of segment
  ray.Align(vector);
  end = Start() + Vector() * length;
}



void R3Segment::
Transform (const R3Matrix& matrix)
{
  // Transform segment
  end.Transform(matrix);
  ray.Transform(matrix);
  length = R3Distance(Start(), End());
}



void R3Segment::
InverseTransform (const R3Matrix& matrix)
{
  // Transform segment
  end.InverseTransform(matrix);
  ray.InverseTransform(matrix);
  length = R3Distance(Start(), End());
}



void R3Segment::
Flip(void)
{
  // Flip direction of segment
  R3Point swap = Start();
  ray.Reposition(end);
  end = swap;
  ray.Flip();
}



void R3Segment::
Translate(const R3Vector& vector)
{
  // Move endpoints of segment
  ray.Translate(vector);
  end += vector;
}



void R3Segment::
Reset(const R3Point& point1, const R3Point& point2)
{
  // Reset segment
  ray = R3Ray(point1, point2);
  end = point2;
  length = R3Distance(point1, point2);
}


const bool R3Segment::
operator==(const R3Segment& segment) const
{
  // Return whether segment is equal
  if (Start() != segment.Start()) return false;
  if (End() != segment.End()) return false;
  return true;
}




void R3Segment::
Draw (void) const
{
  // Draw segment
}



void R3Segment::
Print(FILE *fp) const
{
  // Print end points
  const R3Point& start = Start();
  fprintf(fp, "(%g %g %g) (%g %g %g)", start[0], start[1], start[2], end[0], end[1], end[2]);
}




