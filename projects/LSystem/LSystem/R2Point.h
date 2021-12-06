// Include file for the R2 point class 


// Class definition 

class R2Point {
 public:
  // Constructors
  R2Point(void);
  R2Point(const R2Point& point);
  R2Point(double x, double y);
  R2Point(double array[2]);

  // Coordinate access
  double X(void) const;
  double Y(void) const;
  double operator[](int dim) const;
  double& operator[](int dim);

  // Properties
  bool IsZero(void) const;
  R2Vector Vector(void) const;
  bool operator==(const R2Point& point) const;
  bool operator!=(const R2Point& point) const;

  // Manipulation functions
  void SetX(double x);
  void SetY(double y);
  void SetCoord(int dim, double coord);
  void Translate(const R2Vector& vector);
  void Project(const R2Line& line);
  void Mirror(const R2Line& line);
  void Rotate(const R2Point& origin, double angle);
  void Reset(double x, double y);

  // Assignment operators
  R2Point& operator=(const R2Point& point);
  R2Point& operator+=(const R2Point& point);
  R2Point& operator+=(const R2Vector& vector);
  R2Point& operator-=(const R2Vector& vector);
  R2Point& operator*=(double a);
  R2Point& operator/=(double a);

  // Arithmetic operators
  friend R2Point operator+(const R2Point& point);
  friend R2Point operator-(const R2Point& point);
  friend R2Point operator+(const R2Point& point1, const R2Point& point2);
  friend R2Point operator+(const R2Point& point, const R2Vector& vector);
  friend R2Point operator+(const R2Vector& vector, const R2Point& point);
  friend R2Vector operator-(const R2Point& point1, const R2Point& point2);
  friend R2Point operator-(const R2Point& point, const R2Vector& vector);
  friend R2Point operator*(const R2Point& point, double a);
  friend R2Point operator*(double a, const R2Point& point);
  friend R2Point operator/(const R2Point& point, double a);

  // Output functions
  void Print(FILE *fp = stdout) const;

 private:
  // Internal data
  double v[2];
};



// Public variables

extern R2Point R2null_point;
extern R2Point R2ones_point;
extern R2Point R2posx_point;
extern R2Point R2posy_point;
extern R2Point R2negx_point;
extern R2Point R2negy_point;
extern R2Point R2infinite_point;
#define R2zero_point R2null_point



// Inline functions

inline double R2Point::
X(void) const
{
  // Return X coordinate
  return v[0];
}



inline double R2Point::
Y(void) const
{
  // Return Y coordinate
  return v[1];
}



inline double R2Point::
operator[](int dim) const
{
  // Return X (dim=0) or Y (dim=1) coordinate
  return v[dim];
}



inline double& R2Point::
operator[](int dim)
{
  // Return X (dim=0) or Y (dim=1) coordinate
  return v[dim];
}



inline bool R2Point::
IsZero(void) const
{
  // Return whether point is zero
  return ((v[0] == 0.0) && (v[1] == 0.0));
}



inline void R2Point::
SetX (double x) 
{
  // Set X coord
  v[0] = x;
}



inline void R2Point::
SetY (double y) 
{
  // Set Y coord
  v[1] = y;
}



inline void R2Point::
SetCoord (int dim, double coord) 
{
  // Set coord
  assert ((dim>=0)&&(dim<=1));
  v[dim] = coord;
}



inline void R2Point::
Reset(double x, double y) 
{
  // Set all coords
  v[0] = x;
  v[1] = y;
}



inline void R2Point::
Translate (const R2Vector& vector) 
{
  // Move point by vector
  *this += vector;
}



inline bool R2Point::
operator==(const R2Point& point) const
{
    // Return whether point is equal
    return ((v[0] == point.v[0]) && (v[1] == point.v[1]));
}



inline bool R2Point::
operator!=(const R2Point& point) const
{
    // Return whether point is not equal
    return ((v[0] != point.v[0]) || (v[1] != point.v[1]));
}



