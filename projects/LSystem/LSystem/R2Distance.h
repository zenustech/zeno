// Include file for R2 distance utility 



// Function declarations 

double R2Distance(const R2Point& point1, const R2Point& point2);
double R2Distance(const R2Point& point, const R2Line& line);
double R2Distance(const R2Point& point, const R2Segment& segment);

double R2Distance(const R2Line& line, const R2Point& point);
double R2Distance(const R2Line& line1, const R2Line& line2);
double R2Distance(const R2Line& line, const R2Segment& segment);

double R2Distance(const R2Segment& segment, const R2Point& point);
double R2Distance(const R2Segment& segment, const R2Line& line);
double R2Distance(const R2Segment& segment1, const R2Segment& segment2);

double R2SignedDistance(const R2Point& point, const R2Line& line);
double R2SignedDistance(const R2Line& line, const R2Point& point);



// Inline functions

inline double R2Distance(const R2Line& line, const R2Point& point)
{
  // Distance is commutative
  return R2Distance(point, line);
}



inline double R2Distance(const R2Segment& segment, const R2Point& point)
{
  // Distance is commutative
  return R2Distance(point, segment);
}



inline double R2Distance(const R2Segment& segment, const R2Line& line)
{
  // Distance is commutative
  return R2Distance(line, segment);
}



inline double R2SignedDistance(const R2Point& point, const R2Line& line)
{
  // Distance is commutative
  return R2SignedDistance(line, point);
}



