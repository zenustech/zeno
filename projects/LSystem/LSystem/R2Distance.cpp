// Source file for R2 distance utility 



// Include files 

#include "R2.h"



// Public distance functions 

double R2Distance(const R2Point& point1, const R2Point& point2)
{
  // Return length of vector between points
  R2Vector v = point1 - point2;
  return v.Length();
}



double R2Distance(const R2Point& point, const R2Line& line)
{
  // Return distance from point to line 
  double d = point.X() * line.A() + point.Y() * line.B() + line.C();
  return (d < 0.0) ? -d : d;
}



double R2Distance(const R2Point& point, const R2Segment& segment)
{
  // Check segment
  if (segment.IsPoint()) return R2Distance(point, segment.Start());

  // Check if start point is closest
  R2Vector v1 = point - segment.Start();
  double dir1 = v1.Dot(segment.Vector());
  if (dir1 < 0) return v1.Length();

  // Check if end point is closest
  R2Vector v2 = point - segment.End();
  double dir2 = v2.Dot(segment.Vector());
  if (dir2 > 0) return v2.Length();

  // Return distance from point to segment line
  return R2Distance(point, segment.Line());
}



double R2Distance(const R2Line& line1, const R2Line& line2)
{
  // Return distance from line to line 
  if (line1.Vector() != line2.Vector()) return 0;
  else return fabs(line1.C() - line2.C());
}



double R2Distance(const R2Line& line, const R2Segment& segment)
{
  // Return distance from segment to line
  double d1 = R2SignedDistance(line, segment.Start());
  double d2 = R2SignedDistance(line, segment.End());
  if (d1 < 0) {
    if (d2 < 0) return (d1 > d2) ? -d1 : -d2;
    else return 0;
  }
  else {
    if (d2 > 0) return (d1 < d2) ? d1 : d2;
    else return 0;
  }
}



double R2SignedDistance(const R2Line& line, const R2Point& point)
{
  // Return signed distance from line to point
  return line.A() * point.X() + line.B() * point.Y() + line.C();
}


