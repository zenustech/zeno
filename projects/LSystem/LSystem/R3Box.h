// Include file for the R3 Box class 



// Class definition 

class R3Box {
 public:
  // Constructor functions
  R3Box(void);
  R3Box(const R3Box& box);
  R3Box(const R3Point& min, const R3Point& max);
  R3Box(double xmin, double ymin, double zmin, 
        double xmax, double ymax, double zmax);

  // Box property functions/operators
  R3Point& operator[](int dir);
  const R3Point& operator[](int dir) const;
  const R3Point& Min(void) const;
  const R3Point& Max(void) const;
  double XMin(void) const;
  double YMin(void) const;
  double ZMin(void) const;
  double XMax(void) const;
  double YMax(void) const;
  double ZMax(void) const;
  double Coord(int dir, int dim) const;
  R3Point Corner(int xdir, int ydir, int zdir) const;
  R3Point Centroid(void) const;
  R3Point ClosestPoint(const R3Point& point) const;
  bool IsEmpty(void) const;
  double XLength(void) const;
  double YLength(void) const;
  double ZLength(void) const;
  double AxisLength(const int axis) const;
  double DiagonalLength(void) const;
  double XRadius(void) const;
  double YRadius(void) const;
  double ZRadius(void) const;
  double AxisRadius(const int axis) const;
  double DiagonalRadius(void) const;
  double XCenter(void) const;
  double YCenter(void) const;
  double ZCenter(void) const;
  double AxisCenter(const int axis) const;
  int ShortestAxis(void) const;
  int LongestAxis(void) const;
  double ShortestAxisLength(void) const;
  double LongestAxisLength(void) const;

  // Manipulation functions/operators
  void Empty(void);
  void Translate(const R3Vector& vector);
  void Union(const R3Point& point);
  void Union(const R3Box& box);
  void Union(const R3Sphere& sphere);
  void Intersect(const R3Box& box);
  void Transform(const R3Matrix& matrix);
  void Reset(const R3Point& min, const R3Point& max);
	
  // Relationship functions/operators
  bool operator==(const R3Box& box) const;
  bool operator!=(const R3Box& box) const;

  // Output functions
  void Draw(void) const;
  void Outline(void) const;
  void Print(FILE *fp = stdout) const;

 private:
  R3Point minpt;
  R3Point maxpt;
};



// Public variables 

extern const R3Box R3null_box;
extern const R3Box R3zero_box;
extern const R3Box R3unit_box;



// Inline functions 

inline R3Point& R3Box::
operator[] (int dir) 
{
  // Return min or max point 
  return (dir == 0) ? minpt : maxpt;
}



inline const R3Point& R3Box::
operator[] (int dir) const
{
  // Return min or max point 
  return (dir == 0) ? minpt : maxpt;
}



inline const R3Point& R3Box::
Min (void) const
{
  // Return point (min, min, min)
  return minpt;
}



inline const R3Point& R3Box::
Max (void) const
{
  // Return point (max, max, max)
  return maxpt;
}



inline double R3Box::
XMin (void) const
{
  // Return X coordinate of low corner
  return minpt.X();
}



inline double R3Box::
YMin (void) const
{
  // Return Y coordinate of low corner
  return minpt.Y();
}



inline double R3Box::
ZMin (void) const
{
  // Return Z coordinate of low corner
  return minpt.Z();
}



inline double R3Box::
XMax (void) const
{
  // Return X coordinate of high corner
  return maxpt.X();
}



inline double R3Box::
YMax (void) const
{
  // Return Y coordinate of high corner
  return maxpt.Y();
}



inline double R3Box::
ZMax (void) const
{
  // Return Z coordinate of high corner
  return maxpt.Z();
}



inline double R3Box::
Coord (int dir, int dim) const
{
  // Return requested coordinate 
  return (dir == 0) ? minpt[dim] : maxpt[dim];
}



inline bool R3Box::
IsEmpty (void) const
{
  // Return whether bounding box contains no space
  return ((minpt.X() > maxpt.X()) || (minpt.Y() > maxpt.Y()) || (minpt.Z() > maxpt.Z()));
}



inline double R3Box::
AxisLength (const int axis) const
{
  // Return length of Box along axis
  return (maxpt[axis] - minpt[axis]);
}



inline double R3Box::
XLength (void) const
{
  // Return length in X dimension
  return this->AxisLength(R3_X);
}



inline double R3Box::
YLength (void) const
{
  // Return length in Y dimension
  return this->AxisLength(R3_Y);
}



inline double R3Box::
ZLength (void) const
{
  // Return length in Z dimension
  return this->AxisLength(R3_Z);
}



inline double R3Box::
AxisRadius (const int axis) const
{
  // Return radius of Box along axis
  return 0.5 * (maxpt[axis] - minpt[axis]);
}



inline double R3Box::
XRadius (void) const
{
  // Return radius in X dimension
  return this->AxisRadius(R3_X);
}



inline double R3Box::
YRadius (void) const
{
  // Return radius in Y dimension
  return this->AxisRadius(R3_Y);
}



inline double R3Box::
ZRadius (void) const
{
  // Return radius in Z dimension
  return this->AxisRadius(R3_Z);
}



inline double R3Box::
DiagonalRadius (void) const
{
  // Return radius of Box along diagonal
  return (0.5 * DiagonalLength());
}



inline double R3Box::
AxisCenter (const int axis) const
{
  // Return radius of Box along axis
  return 0.5 * (maxpt[axis] + minpt[axis]);
}



inline double R3Box::
XCenter (void) const
{
  // Return center in X dimension
  return this->AxisCenter(R3_X);
}



inline double R3Box::
YCenter (void) const
{
  // Return center in Y dimension
  return this->AxisCenter(R3_Y);
}



inline double R3Box::
ZCenter (void) const
{
  // Return center in Z dimension
  return this->AxisCenter(R3_Z);
}



inline bool R3Box::
operator==(const R3Box& box) const
{
  // Return whether box is equal
  return ((minpt == box.minpt) && (maxpt == box.maxpt));
}



inline bool R3Box::
operator!=(const R3Box& box) const
{
  // Return whether box is not equal
  return (!(*this == box));
}



