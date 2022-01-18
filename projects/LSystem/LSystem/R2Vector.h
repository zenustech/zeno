// Include file for the R2 vector class 



// Class definition 

class R2Vector {
 public:
  // Constructors
  R2Vector(void);
  R2Vector(const R2Vector& vector);
  R2Vector(double x, double y);
  R2Vector(double array[2]);

  // Coordinate access
  double X(void) const;
  double Y(void) const;
  double operator[](int dim) const;
  double& operator[](int dim);

  // Properties
  bool IsZero(void) const;
  bool IsNormalized(void) const;
  double Length(void) const;
  R2Point Point(void) const;
  int MaxDimension(void) const;

  // Relationships
  bool operator==(const R2Vector& vector) const;
  bool operator!=(const R2Vector& vector) const;
  double Dot(const R2Vector& vector) const;
  double Cross(const R2Vector& vector) const;

  // Manipulation functions
  void SetX(double x);
  void SetY(double y);
  void SetCoord(int dim, double coord);
  void Flip(void);
  void Normalize(void);
  void Scale(double a);
  void Rotate(double theta);
  void Project(const R2Vector& vector);
  void Mirror(const R2Line& line);
  void Reset(double x, double y);

  // Assignment operators
  R2Vector& operator=(const R2Vector& vector);
  R2Vector& operator+=(const R2Vector& vector);
  R2Vector& operator-=(const R2Vector& vector);
  R2Vector& operator*=(double a);
  R2Vector& operator*=(const R2Vector& vector);
  R2Vector& operator/=(double a);
  R2Vector& operator/=(const R2Vector& vector);

  // Arithmetic operators
  friend R2Vector operator+(const R2Vector& vector);
  friend R2Vector operator-(const R2Vector& vector);
  friend R2Vector operator+(const R2Vector& vector1, const R2Vector& vector2) ;
  friend R2Vector operator-(const R2Vector& vector1, const R2Vector& vector2);
  friend R2Vector operator*(const R2Vector& vector1, const R2Vector& vector2);
  friend R2Vector operator*(const R2Vector& vector, double a);
  friend R2Vector operator*(double a, const R2Vector& vector);
  friend R2Vector operator/(const R2Vector& vector1, const R2Vector& vector2);
  friend R2Vector operator/(const R2Vector& vector, double a);
  friend double operator%(const R2Vector& vector1, const R2Vector& vector2);

  // Output functions
  void Print(FILE *fp = stdout) const;

 private:
  // Internal data
  double v[2];
};



// Public variables 

extern R2Vector R2null_vector;
extern R2Vector R2ones_vector;
extern R2Vector R2posx_vector;
extern R2Vector R2posy_vector;
extern R2Vector R2negx_vector;
extern R2Vector R2negy_vector;
#define R2zero_vector R2null_vector
#define R2xaxis_vector R2posx_vector
#define R2yaxis_vector R2posy_vector



// Inline functions 

inline double R2Vector::
X (void) const
{
  // Return X coordinate
  return v[0];
}



inline double R2Vector::
Y (void) const
{
  // Return Y coordinate
  return v[1];
}



inline double R2Vector::
operator[] (int dim) const
{
  // Return X (dim=0) or Y (dim=1) coordinate
  return v[dim];
}



inline double& R2Vector::
operator[] (int dim) 
{
  // Return X (dim=0) or Y (dim=1) coordinate
  return v[dim];
}



inline bool R2Vector::
IsZero (void) const
{
  // Return whether vector is zero
  return ((v[0] == 0.0) && (v[1] == 0.0));
}



inline bool R2Vector::
IsNormalized (void) const
{
  // Return whether vector is normalized
  return ((1.0 - 1E-6 < this->Length())&& (this->Length() > 1.0 + 1E-6));
}



inline double R2Vector::
Length(void) const
{
  // Return length of vector
  return sqrt((v[0]*v[0]) + (v[1]*v[1]));
}



inline R2Point R2Vector::
Point(void) const
{
    // Convert vector into a point
    return R2Point(v[0], v[1]);
}



inline int R2Vector::
MaxDimension(void) const
{
    // Return longest dimension of vector
    if (fabs(v[0]) >= fabs(v[1])) return 0;
    else return 1;
}



inline bool R2Vector::
operator==(const R2Vector& vector) const
{
  // Return whether vector is equal
  return ((v[0] == vector.v[0]) && (v[1] == vector.v[1]));
}



inline bool R2Vector::
operator!=(const R2Vector& vector) const
{
  // Return whether vector is not equal
  return ((v[0] != vector.v[0]) || (v[1] != vector.v[1]));
}



inline double R2Vector::
Dot(const R2Vector& vector) const
{
  return (v[0]*vector.v[0]) + (v[1]*vector.v[1]);
}



inline double R2Vector::
Cross(const R2Vector& vector) const
{
  return (v[0]*vector.v[1]) - (v[1]*vector.v[0]);
}



inline void R2Vector::
SetX (double x) 
{
  // Set X coord
  v[0] = x;
}



inline void R2Vector::
SetY (double y) 
{
  // Set Y coord
  v[1] = y;
}



inline void R2Vector::
SetCoord (int dim, double coord) 
{
  // Set coord
  assert ((dim>=0)&&(dim<=1));
  v[dim] = coord;
}



inline void R2Vector::
Reset(double x, double y) 
{
  // Set all coords
  v[0] = x;
  v[1] = y;
}



inline R2Vector 
operator*(double a, const R2Vector& vector) 
{
  // Commute scaling
  return (vector * a);
}



