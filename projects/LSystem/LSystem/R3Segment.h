// Include file for the R3 segment class 



// Class definition 

class R3Segment {
 public:
  // Constructor functions
  R3Segment(void);
  R3Segment(const R3Segment& segment);
  R3Segment(const R3Point& point, const R3Vector& vector);
  R3Segment(const R3Point& point1, const R3Point& point2);
  R3Segment(double x1, double y1, double z1, double x2, double y2, double z2);

  // Property functions/operators
  const R3Point& Start(void) const;
  const R3Point& End(void) const;
  const R3Vector& Vector(void) const;
  const R3Point Point(int k) const;
  const R3Point Point(double t) const;
  const R3Point& operator[](int k) const;
  const R3Ray& Ray(void) const;
  const R3Line& Line(void) const;
  const R3Point Midpoint(void) const;
  const R3Point Centroid(void) const;
  const R3Box BBox(void) const;
  const double Length(void) const;
  const double T(const R3Point& point) const;
  const bool IsPoint(void) const;
  const bool operator==(const R3Segment& segment) const;
  const bool operator!=(const R3Segment& segment) const;

  // Manipulation functions/operators
  void Flip(void);
  void Mirror(const R3Plane& plane);
  void Translate(const R3Vector& vector);
  void Reposition(int k, const R3Point& point);
  void Align(const R3Vector& vector);
  void Transform(const R3Matrix& matrix);
  void InverseTransform(const R3Matrix& matrix);
  void Reset(const R3Point& point1, const R3Point& point2);

  // Arithmetic functions/operators
  R3Segment operator-(void) const;

  // Output functions
  void Draw(void) const;
  void Print(FILE *fp = stdout) const;

 private:
  R3Ray ray;
  R3Point end;
  double length;
};



// Public variables 

extern R3Segment R3null_segment;




// Inline functions 

inline const R3Point& R3Segment::
Start (void) const
{
  // Return start point of segment
  return ray.Start();
}



inline const R3Point& R3Segment::
End (void) const
{
  // Return end point of segment
  return end;
}



inline const R3Vector& R3Segment::
Vector(void) const
{
  // Return direction vector of segment 
  return ray.Vector();
}



inline const R3Point& R3Segment::
operator[] (int k) const
{
  // Return kth endpoint of segment
  assert((k>=0)&&(k<=1));
  return (k==0) ? Start() : End();
}



inline const R3Point R3Segment::
Point (int k) const
{
  // Return kth endpoint of segment
  return (*this)[k];
}



inline const R3Ray& R3Segment::
Ray(void) const
{
  // Return ray along segment
  return ray;
}



inline const R3Line& R3Segment::
Line(void) const
{
  // Return line containing segment
  return ray.Line();
}



inline const R3Point R3Segment::
Centroid(void) const
{
  // Return midpoint of segment
  return Midpoint();
}



inline const double R3Segment::
Length(void) const
{
  // Return length of segment
  return length;
}



inline const bool R3Segment::
operator!=(const R3Segment& segment) const
{
  // Return whether segment is not equal
  return (!(*this == segment));
}



inline R3Segment R3Segment::
operator-(void) const
{
  // Return segment with flipped orientation
  return R3Segment(End(), Start());
}



inline void R3Segment::
Mirror(const R3Plane& plane)
{
  // Mirror segment over plane
  ray.Mirror(plane);
  end.Mirror(plane);
}





