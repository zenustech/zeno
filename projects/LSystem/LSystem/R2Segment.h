// Include file for the R2 segment class


// Class definition 

class R2Segment {
 public:
  // Constructors
  R2Segment(void);
  R2Segment(const R2Segment& segment);
  R2Segment(const R2Point& point, const R2Vector& vector);
  R2Segment(const R2Point& point1, const R2Point& point2);
  R2Segment(double x1, double y1, double x2, double y2);

  // Properties
  R2Point Start(void) const;
  R2Point End(void) const;
  R2Point Point(int k) const;
  R2Point operator[](int k) const;
  R2Vector Vector(void) const;
  R2Vector Normal(void) const;
  R2Point Midpoint(void) const;
  R2Point Point(double t) const;
  R2Line Line(void) const;
  double Length(void) const;
  double T(const R2Point& point) const;
  bool IsPoint(void) const;
  bool operator==(const R2Segment& segment) const;
  bool operator!=(const R2Segment& segment) const;

  // Manipulation functions
  void Flip(void);
  void Project(const R2Line& line);
  void Mirror(const R2Line& line);
  void Translate(const R2Vector& vector);
  void Rotate(const R2Point& origin, double angle);
  void Reset(const R2Point& point1, const R2Point& point2);
  void SetStart(const R2Point& point);
  void SetEnd(const R2Point& point);
  void SetPoint(const R2Point& point, int k);

  // Arithmetic operators
  R2Segment operator-(void);

  // Print functions
  void Print(FILE *fp = stdout) const;
	
 private:
  // Internal data
  R2Line line;
  R2Point points[2];
  double length;
};



/* Inline functions */

inline R2Point R2Segment::
operator[] (int k) const
{
  // Return kth endpoint of segment
  assert((k>=0)&&(k<=1));
  return points[k];
}



inline R2Point R2Segment::
Point (int k) const
{
  // Return kth endpoint of segment
  assert((k>=0)&&(k<=1));
  return points[k];
}



inline R2Point R2Segment::
Start (void) const
{
  // Return start point of segment
  return Point(0);
}



inline R2Point R2Segment::
End (void) const
{
  // Return end point of segment
  return Point(1);
}



inline R2Vector R2Segment::
Vector(void) const
{
  // Return direction vector of segment 
  return line.Vector();
}



inline R2Vector R2Segment::
Normal(void) const
{
  // Return normal vector of segment 
  return line.Normal();
}



inline R2Point R2Segment::
Point(double t) const
{
  // Return point along segment
  return (Start() + Vector() * t);
}



inline R2Point R2Segment::
Midpoint(void) const
{
  // Return midpoint of segment
  return (Start() + End()) * 0.5;
}



inline R2Line R2Segment::
Line(void) const
{
  // Return line containing segment
  return line;
}



inline double R2Segment::
Length(void) const
{
  // Return length of segment
  return length;
}



inline double R2Segment::
T(const R2Point& point) const
{
  // Return parametric value of closest point on segment
  double denom = Vector().Dot(Vector());
  if (denom == 0) return 0.0;
  R2Vector topoint = point - Start();
  return (Vector().Dot(topoint) / denom);
}



inline bool R2Segment::
IsPoint(void) const
{
  // Return whether segment covers a single point
  return (length == 0.0);
}



inline bool R2Segment::
operator==(const R2Segment& segment) const
{
    // Return whether segment is equal
    if (Start() != segment.Start()) return false;
    if (End() != segment.End()) return false;
    return true;
}


inline bool R2Segment::
operator!=(const R2Segment& segment) const
{
  // Return whether segment is not equal
  return (!(*this == segment));
}



inline void R2Segment::
SetStart(const R2Point& point)
{
  // Set the start position
  SetPoint(point, 0);
}



inline void R2Segment::
SetEnd(const R2Point& point)
{
  // Set the end position
  SetPoint(point, 1);
}



