// Include file for the R3 point class 



// Class definition 

class R3Point {
 public:
  // Constructors
  R3Point(void);
  R3Point(const R3Point& point);
  R3Point(double x, double y, double z);
  R3Point(const double array[3]);

  // Coordinate access
  double X(void) const;
  double Y(void) const;
  double Z(void) const;
  double operator[](int dim) const;
  double& operator[](int dim);

  // Properties
  bool IsZero(void) const;
  R3Vector Vector(void) const;
  bool operator==(const R3Point& point) const;
  bool operator!=(const R3Point& point) const;

  // Manipulation functions
  void SetX(double x);
  void SetY(double y);
  void SetZ(double z);
  void SetCoord(int dim, double coord);
  void Translate(const R3Vector& vector);
  void Project(const R3Line& line);
  void Project(const R3Plane& plane);
  void Mirror(const R3Plane& plane);
  void Rotate(const R3Vector& axis, double theta);
  void Rotate(const R3Line& axis, double theta);
  void Transform(const R3Matrix& matrix);
  void InverseTransform(const R3Matrix& matrix);
  void Reset(double x, double y, double z);

  // Assignment operators
  R3Point& operator=(const R3Point& point);
  R3Point& operator+=(const R3Point& point);
  R3Point& operator+=(const R3Vector& vector);
  R3Point& operator-=(const R3Vector& vector);
  R3Point& operator*=(const double a);
  R3Point& operator/=(const double a);

  // Arithmetic operators
  friend R3Point operator-(const R3Point& point);
  friend R3Point operator+(const R3Point& point1, const R3Point& point2);
  friend R3Point operator+(const R3Point& point, const R3Vector& vector);
  friend R3Point operator+(const R3Vector& vector, const R3Point& point);
  friend R3Vector operator-(const R3Point& point1, const R3Point& point2);
  friend R3Point operator-(const R3Point& point, const R3Vector& vector);
  friend R3Point operator*(const R3Point& point, const double a);
  friend R3Point operator*(const double a, const R3Point& point);
  friend R3Point operator/(const R3Point& point, const double a);

  // Output functions
  void Draw(void) const;
  void Print(FILE *fp = stdout) const;

 private:
  double v[3];
};



// Public variables 

extern const R3Point R3null_point;
extern const R3Point R3ones_point;
extern const R3Point R3posx_point;
extern const R3Point R3posy_point;
extern const R3Point R3posz_point;
extern const R3Point R3negx_point;
extern const R3Point R3negy_point;
extern const R3Point R3negz_point;
#define R3zero_point R3null_point



// Inline functions 

inline double R3Point::
X (void) const
{
  // Return X coordinate
  return(v[0]);
}



inline double R3Point::
Y (void) const
{
  // Return y coordinate
  return(v[1]);
}



inline double R3Point::
Z (void) const
{
  // Return z coordinate
  return(v[2]);
}



inline double R3Point::
operator[](int dim) const
{
  // Return coordinate in given dimension (0=X, 1=Y, 2=Z)
  assert((dim>=R3_X) && (dim<=R3_Z));
  return(v[dim]);
}



inline double& R3Point::
operator[] (int dim) 
{
  // Return reference to coordinate in given dimension (0=X, 1=Y, 2=Z)
  assert((dim>=R3_X) && (dim<=R3_Z));
  return(v[dim]);
}



inline bool R3Point::
IsZero(void) const
{
  // Return whether point is zero
  return ((v[0] == 0.0) && (v[1] == 0.0) && (v[2] == 0.0));
}



inline void R3Point::
SetX (double x) 
{
  // Set X coord
  v[0] = x;
}



inline void R3Point::
SetY (double y) 
{
  // Set Y coord
  v[1] = y;
}



inline void R3Point::
SetZ (double z) 
{
  // Set Z coord
  v[2] = z;
}



inline void R3Point::
SetCoord (int dim, double coord) 
{
  // Set coord in given dimension
  v[dim] = coord;
}



inline void R3Point::
Reset(double x, double y, double z) 
{
  // Set all coords
  v[0] = x;
  v[1] = y;
  v[2] = z;
}



inline bool R3Point::
operator==(const R3Point& point) const
{
  // Return whether point is equal
  return ((v[0] == point.v[0]) && (v[1] == point.v[1]) && (v[2] == point.v[2]));
}



inline bool R3Point::
operator!=(const R3Point& point) const
{
  // Return whether point is not equal
  return ((v[0] != point.v[0]) || (v[1] != point.v[1]) || (v[2] != point.v[2]));
}



