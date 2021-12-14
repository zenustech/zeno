// Source file for R3 distance utility 



// Include files 

#include "R3.h"



// Public functions 

double R3SquaredDistance(const R3Point& point1, const R3Point& point2)
{
  // Return length of vector between points
  R3Vector v = point1 - point2;
  return v.Dot(v);
}



double R3Distance(const R3Point& point1, const R3Point& point2)
{
  // Return length of vector between points
  R3Vector v = point1 - point2;
  return v.Length();
}



double R3Distance(const R3Point& point, const R3Line& line)
{
  // Return distance from point to line (Riddle p. 904)
  R3Vector v = line.Vector();
  v.Cross(point - line.Point());
  return v.Length();
}



double R3Distance(const R3Point& point, const R3Ray& ray)
{
  // Check if start point is closest
  R3Vector v = point - ray.Start();
  double dir = v.Dot(ray.Vector());
  if (dir < 0) return v.Length();

  // Return distance from point to ray line
  return R3Distance(point, ray.Line());
}



double R3Distance(const R3Point& point, const R3Segment& segment)
{
  // Check if start point is closest
  R3Vector v1 = point - segment.Start();
  double dir1 = v1.Dot(segment.Vector());
  if (dir1 < 0) return v1.Length();

  // Check if end point is closest
  R3Vector v2 = point - segment.End();
  double dir2 = v2.Dot(segment.Vector());
  if (dir2 < 0) return v2.Length();

  // Return distance from point to segment line
  return R3Distance(point, segment.Line());
}



double R3Distance(const R3Point& point, const R3Plane& plane)
{
  // Return distance from point to plane
  double d = R3SignedDistance(plane, point);
  if (d > 0) return d;
  else if (d < 0) return -d;
  else return 0.0;
}



double R3Distance(const R3Point& point, const R3Box& box)
{
  // Find axial distances from point to box
  double dx, dy, dz;
  if (point.X() > box.XMax()) dx = point.X() - box.XMax();
  else if (point.X() < box.XMin()) dx = box.XMin()- point.X();
  else dx = 0.0;
  if (point.Y() > box.YMax()) dy = point.Y() - box.YMax();
  else if (point.Y() < box.YMin()) dy = box.YMin()- point.Y();
  else dy = 0.0;
  if (point.Z() > box.ZMax()) dz = point.Z() - box.ZMax();
  else if (point.Z() < box.ZMin()) dz = box.ZMin()- point.Z();
  else dz = 0.0;
    
  // Return distance between point and closest point in box 
  if ((dy == 0.0) && (dz == 0.0)) return dx;
  else if ((dx == 0.0) && (dz == 0.0)) return dy;
  else if ((dx == 0.0) && (dy == 0.0)) return dz;
  else return sqrt(dx*dx + dy*dy + dz*dz);
}



double R3Distance(const R3Line& line1, const R3Line& line2)
{
  // Return distance from line to line (Riddle p. 905)
  R3Vector v = line1.Vector();
  v.Cross(line2.Vector());
  return v.Dot(line1.Point() - line2.Point());
}



double R3Distance(const R3Line& line, const R3Ray& ray)
{
  fprintf(stderr, "Not implemented");
  return 0.0;
}



double R3Distance(const R3Line& line, const R3Segment& segment)
{
  fprintf(stderr, "Not implemented");
  return 0.0;
}



double R3Distance(const R3Line& line, const R3Plane& plane)
{
  // Return distance from line to plane
  double d = R3SignedDistance(plane, line);
  if (d > 0) return d;
  else if (d < 0) return -d;
  else return 0.0;
}



double R3Distance(const R3Line& line, const R3Box& box)
{
  fprintf(stderr, "Not implemented");
  return 0.0;
}



double R3Distance(const R3Ray& ray1, const R3Ray& ray2)
{
  fprintf(stderr, "Not implemented");
  return 0.0;
}



double R3Distance(const R3Ray& ray, const R3Segment& segment)
{
  // There's got to be a better way ???

  // Get vectors in more convenient form
  const R3Vector v1 = ray.Vector();
  const R3Vector v2 = segment.Vector();

  // Compute useful intermediate values
  const double v1v1 = 1.0;  // v1.Dot(v1);
  const double v2v2 = 1.0;  // v2.Dot(v2);
  double v1v2 = v1.Dot(v2);
  double denom = v1v2*v1v2 - v1v1*v2v2;

  // Check if ray and segment are parallel
  if (denom == 0) {
    // Not right ???
    // Look at directions of vectors, then check relative starts and stops
    return R3Distance(segment.Line(), ray.Line());
  }
  else {
    // Find closest points
    const R3Vector p1 = ray.Start().Vector();
    const R3Vector p2 = segment.Start().Vector();
    double p1v1 = v1.Dot(p1);
    double p2v2 = v2.Dot(p2);
    double p1v2 = v2.Dot(p1);
    double p2v1 = v1.Dot(p2);
    double ray_t = (v1v2*p2v2 + v2v2*p1v1 - v1v2*p1v2 - v2v2*p2v1) / denom;
    double segment_t = (v1v2*p1v1 + v1v1*p2v2 - v1v2*p2v1 - v1v1*p1v2) / denom;
    R3Point ray_point = (ray_t <= 0.0) ? ray.Start() : ray.Point(ray_t);
    R3Point segment_point = (segment_t <= 0.0) ? segment.Start() : 
      (segment_t >= segment.Length()) ? segment.End() : segment.Ray().Point(segment_t);
    double distance = R3Distance(ray_point, segment_point);
    return distance;
  }
}



double R3Distance(const R3Ray& ray, const R3Plane& plane)
{
  // Return distance from ray to plane
  double d = R3SignedDistance(plane, ray);
  if (d > 0) return d;
  else if (d < 0) return -d;
  else return 0.0;
}



double R3Distance(const R3Ray& ray, const R3Box& box)
{
  fprintf(stderr, "Not implemented");
  return 0.0;
}



double R3Distance(const R3Segment& segment, const R3Plane& plane)
{
  // Return distance from segment to plane
  double d = R3SignedDistance(plane, segment);
  if (d > 0) return d;
  else if (d < 0) return -d;
  else return 0.0;
}



double R3Distance(const R3Segment& segment, const R3Box& box)
{
  fprintf(stderr, "Not implemented");
  return 0.0;
}



double R3Distance(const R3Plane& plane1, const R3Plane& plane2)
{
  // Return distance from plane to plane
  double d = R3SignedDistance(plane1, plane2);
  if (d > 0) return d;
  else if (d < 0) return -d;
  else return 0.0;
}



double R3Distance(const R3Plane& plane, const R3Box& box)
{
  // Return distance from plane to box
  double d = R3SignedDistance(plane, box);
  if (d > 0) return d;
  else if (d < 0) return -d;
  else return 0.0;
}



double R3SignedDistance(const R3Plane& plane, const R3Point& point)
{
  // Return signed distance from point to plane (Riddle p. 914)
  return (point.X()*plane.A() + point.Y()*plane.B() + point.Z()*plane.C() + plane.D());
}



double R3SignedDistance(const R3Plane& plane, const R3Line& line)
{
  // Return signed distance from plane to line
  if (plane.Normal().Dot(line.Vector()) == 0) {
    // Plane and line are parallel
    return R3SignedDistance(plane, line.Point());
  }
  else {
    // Plane and line are not parallel
    return 0.0;
  }
}



double R3SignedDistance(const R3Plane& plane, const R3Ray& ray)
{
  // Return signed distance from plane to ray
  double d1 = R3SignedDistance(plane, ray.Start());
  if (d1 > 0) {
    // Start point is above plane
    double dot = ray.Vector().Dot(plane.Normal());
    if (dot < 0) return 0.0;
    else return d1;
  }
  else if (d1 < 0) {
    // Start point is below plane
    double dot = ray.Vector().Dot(plane.Normal());
    if (dot > 0) return 0.0;
    else return d1;
  }
  else {
    // Start point is on plane
    return 0.0;
  }
}



double R3SignedDistance(const R3Plane& plane, const R3Segment& segment)
{
  // Return signed distance from plane to segment
  double d1 = R3SignedDistance(plane, segment.Start());
  if (d1 > 0) {
    // Start point is above plane
    double d2 = R3SignedDistance(plane, segment.End());
    if (d2 > 0) return ((d1 > d2) ? d2 : d1);
    else return 0.0;
  }
  else if (d1 < 0) {
    // Start point is below plane
    double d2 = R3SignedDistance(plane, segment.End());
    if (d2 < 0) return ((d1 > d2) ? d1 : d2);
    else return 0.0;
  }
  else {
    // Start point is on plane
    return 0.0;
  }
}



double R3SignedDistance(const R3Plane& plane1, const R3Plane& plane2)
{
  // Return signed distance from plane to plane
  double dot = plane1.Normal().Dot(plane2.Normal());
  if (dot == 1.0) return (plane1.D() - plane2.D());
  else if (dot == -1.0) return (plane1.D() + plane2.D());
  else return 0.0;
}



double R3SignedDistance(const R3Plane& plane, const R3Box& box)
{
  // Return signed distance from plane to box
  int ix = (plane.Normal().X() > 0) ? 0 : 1;
  int iy = (plane.Normal().Y() > 0) ? 0 : 1;
  int iz = (plane.Normal().Z() > 0) ? 0 : 1;
  double d1 = R3SignedDistance(plane, box.Corner(ix, iy, iz));
  if (d1 >= 0) return d1;
  double d2 = R3SignedDistance(plane, box.Corner(1-ix, 1-iy, 1-iz));
  if (d2 < 0) return d2;
  else return 0.0;
}



