// Source file for the R2 segment class 



// Include files 

#include "R2.h"



R2Segment::
R2Segment(void)
{
}



R2Segment::
R2Segment(const R2Segment& segment)
  : line(segment.line),
    length(segment.length)
{
  points[0] = segment.points[0];
  points[1] = segment.points[1];
}



R2Segment::
R2Segment(const R2Point& point, const R2Vector& vector)
{
  Reset(point, point + vector);
}



R2Segment::
R2Segment(const R2Point& point1, const R2Point& point2)
{
  Reset(point1, point2);
}



R2Segment::
R2Segment(double x1, double y1, double x2, double y2)
{
  Reset(R2Point(x1,y1), R2Point(x2,y2));
}



void R2Segment::
Project(const R2Line& projection_line)
{
  // Project segment onto line
  points[0].Project(projection_line);
  points[1].Project(projection_line);
  R2Vector v(points[1] - points[0]);
  length = v.Length();
  if (length > 0) v /= length;
  line.Reset(points[0], v, true);
}



void R2Segment::
Flip(void)
{
  // Flip direction of segment
  R2Point swap = points[0];
  points[0] = points[1];
  points[1] = swap;
  line.Flip();
}



void R2Segment::
Mirror(const R2Line& mirror_line)
{
  // Mirror segment over line
  points[0].Mirror(mirror_line);
  points[1].Mirror(mirror_line);
  line.Mirror(mirror_line);
}



void R2Segment::
Translate(const R2Vector& vector)
{
  // Translate segment
  points[0].Translate(vector);
  points[1].Translate(vector);
  line.Translate(vector);
}



void R2Segment::
Rotate(const R2Point& origin, double angle)
{
  // Rotate segment around specified origin counterclockwise by angle
  points[0].Rotate(origin, angle);
  points[1].Rotate(origin, angle);
  Reset(points[0], points[1]);
}



void R2Segment::
Reset(const R2Point& point1, const R2Point& point2)
{
  // Reset segment
  points[0] = point1;
  points[1] = point2;
  R2Vector v(points[1] - points[0]);
  length = v.Length();
  if (length > 0) v /= length;
  line.Reset(points[0], v, true);
}



void R2Segment::
SetPoint(const R2Point& point, int k)
{
  // Reset segment
  points[k] = point;
  R2Vector v(points[1] - points[0]);
  length = v.Length();
  if (length > 0) v /= length;
  line.Reset(points[0], v, true);
}



R2Segment R2Segment::
operator-(void)
{
  // Return segment with flipped orientation
  return R2Segment(End(), Start());
}



void R2Segment::
Print(FILE *fp) const
{
  // Print segment endpoints
  points[0].Print(fp);
  fprintf(fp, "  ");
  points[1].Print(fp);
}
