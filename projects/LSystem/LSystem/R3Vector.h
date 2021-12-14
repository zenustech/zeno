// Include file for the R3 vector class 



// Class definition 

class R3Vector {
 public:
  // Constructors
  R3Vector(void);
  R3Vector(const R3Vector& vector);
  R3Vector(double x, double y, double z);
  R3Vector(const double array[3]);
  R3Vector(double pitch, double yaw);

  // Coordinate access
  double X(void) const;
  double Y(void) const;
  double Z(void) const;
  double& operator[](int dim);
  double operator[](int dim) const;

  // Properties
  bool IsZero(void) const;
  bool IsNormalized(void) const;
  double Length(void) const;
  R3Point Point(void) const;
  int MinDimension(void) const;
  int MaxDimension(void) const;

  // Relationships
  bool operator==(const R3Vector& vector) const;
  bool operator!=(const R3Vector& vector) const;
  double Dot(const R3Vector& vector) const;

  // Manipulation functions
  void SetX(double x);
  void SetY(double y);
  void SetZ(double z);
  void SetCoord(int dim, double coord);
  void Flip(void);
  void Normalize(void);
  void Cross(const R3Vector& vector);
  void Rotate(const R3Vector& axis, double theta);
  void Project(const R3Vector& vector);
  void Project(const R3Plane& plane);
  void Mirror(const R3Plane& plane);
  void Transform(const R3Matrix& matrix);
  void InverseTransform(const R3Matrix& matrix);
  void Reset(double x, double y, double z);

  // Assignment operators
  R3Vector& operator=(const R3Vector& vector);
  R3Vector& operator+=(const R3Vector& vector);
  R3Vector& operator-=(const R3Vector& vector);
  R3Vector& operator*=(const double a);
  R3Vector& operator*=(const R3Vector& vector);
  R3Vector& operator/=(const double a);
  R3Vector& operator/=(const R3Vector& vector);

  // Arithmetic operators
  friend R3Vector operator+(const R3Vector& vector);
  friend R3Vector operator-(const R3Vector& vector);
  friend R3Vector operator+(const R3Vector& vector1, const R3Vector& vector2);
  friend R3Vector operator-(const R3Vector& vector1, const R3Vector& vector2);
  friend R3Vector operator*(const R3Vector& vector1, const R3Vector& vector2);
  friend R3Vector operator*(const R3Vector& vector, const double a);
  friend R3Vector operator*(const double a, const R3Vector& vector);
  friend R3Vector operator/(const R3Vector& vector1, const R3Vector& vector2);
  friend R3Vector operator/(const R3Vector& vector, const double a);
  friend R3Vector operator%(const R3Vector& vector1, const R3Vector& vector2);

  // Output functions
  void Draw(void) const;
  void Print(FILE *fp = stdout) const;

 private:
  double v[3];
};



// Public variables 

extern const R3Vector R3null_vector;
extern const R3Vector R3ones_vector;
extern const R3Vector R3posx_vector;
extern const R3Vector R3posy_vector;
extern const R3Vector R3posz_vector;
extern const R3Vector R3negx_vector;
extern const R3Vector R3negy_vector;
extern const R3Vector R3negz_vector;
#define R3zero_vector R3null_vector
#define R3xaxis_vector R3posx_vector
#define R3yaxis_vector R3posy_vector
#define R3zaxis_vector R3posz_vector



// Inline functions 

inline double R3Vector::
X (void) const
{
  // Return X coordinate
  return (v[0]);
}



inline double R3Vector::
Y (void) const
{
  // Return Y coordinate
  return (v[1]);
}



inline double R3Vector::
Z (void) const
{
  // Return Z coordinate
  return (v[2]);
}



inline double R3Vector::
operator[](int dim) const
{
  // Return coordinate in given dimension
  assert((dim>=R3_X) && (dim<=R3_Z));
  return(v[dim]);
}



inline double& R3Vector::
operator[] (int dim) 
{
  // Return reference to coordinate in given dimension
  assert((dim>=R3_X) && (dim<=R3_Z));
  return(v[dim]);
}



inline bool R3Vector::
IsZero (void) const
{
  // Return whether vector is zero
  return ((v[0] == 0.0) && (v[1] == 0.0) && (v[2] == 0.0));
}



inline bool R3Vector::
IsNormalized (void) const
{
  // Return whether vector is normalized
  double length = Length();
  if (length < 1.0 - 1E-6) return false;
  if (length > 1.0 + 1E-6) return false;
  return true;
}



inline double R3Vector::
Length(void) const
{
  // Return length of vector
  return (sqrt((v[0]*v[0]) + (v[1]*v[1]) + (v[2]*v[2])));
}



inline R3Point R3Vector::
Point(void) const
{
  // Return point at (0,0,0) plus vector
  return R3Point(v[0], v[1], v[2]);
}



inline int R3Vector::
MinDimension(void) const
{
  // Return principal dimension of vector
  if (fabs(v[0]) <= fabs(v[1])) {
    if (fabs(v[0]) <= fabs(v[2])) return R3_X;
    else return R3_Z;
  }
  else {
    if (fabs(v[1]) <= fabs(v[2])) return R3_Y;
    else return R3_Z;
  }
}



inline int R3Vector::
MaxDimension(void) const
{
  // Return principal dimension of vector
  if (fabs(v[0]) >= fabs(v[1])) {
    if (fabs(v[0]) >= fabs(v[2])) return R3_X;
    else return R3_Z;
  }
  else {
    if (fabs(v[1]) >= fabs(v[2])) return R3_Y;
    else return R3_Z;
  }
}



inline bool R3Vector::
operator==(const R3Vector& vector) const
{
  // Return whether vector is equal
  return ((v[0] == vector.v[0]) && (v[1] == vector.v[1]) && (v[2] == vector.v[2]));
}



inline bool R3Vector::
operator!=(const R3Vector& vector) const
{
  // Return whether vector is not equal
  return ((v[0] != vector.v[0]) || (v[1] != vector.v[1]) || (v[2] != vector.v[2]));
}



inline void R3Vector::
SetX (double x) 
{
  // Set X coord
  v[0] = x;
}



inline void R3Vector::
SetY (double y) 
{
  // Set Y coord
  v[1] = y;
}



inline void R3Vector::
SetZ (double z) 
{
  // Set Z coord
  v[2] = z;
}



inline void R3Vector::
SetCoord (int dim, double coord) 
{
  // Set coord in given dimension
  v[dim] = coord;
}



inline void R3Vector::
Reset(double x, double y, double z) 
{
  // Set all coords
  v[0] = x;
  v[1] = y;
  v[2] = z;
}



