// Include file for the R3 plane class 



// Class definition 

class R3Plane {
 public:
  // Constructor functions
  R3Plane(void);
  R3Plane(const R3Plane& plane);
  R3Plane(double a, double b, double c, double d);
  R3Plane(const double array[4]);
  R3Plane(const R3Vector& normal, double d);
  R3Plane(const R3Point& point, const R3Vector& normal);
  R3Plane(const R3Point& point, const R3Line& line);
  R3Plane(const R3Point& point, const R3Vector& vector1, const R3Vector& vector2);
  R3Plane(const R3Point& point1, const R3Point& point2, const R3Point& point3);
  R3Plane(const R3Point *points, int npoints);

  // Property functions/operators
  double& operator[](int i);
  double operator[](int i) const;
  double A(void) const;
  double B(void) const;
  double C(void) const;
  double D(void) const;
  R3Point Point(void) const;
  const R3Vector& Normal(void) const;
  bool IsZero(void) const;
  bool operator==(const R3Plane& plane) const;
  bool operator!=(const R3Plane& plane) const;

  // Manipulation functions/operators
  void Flip(void);
  void Mirror(const R3Plane& plane);
  void Translate(const R3Vector& vector);
  void Reposition(const R3Point& point);
  void Align(const R3Vector& normal);
  void Transform(const R3Matrix& matrix);
  void InverseTransform(const R3Matrix& matrix);
  void Reset(const R3Point& point, const R3Vector& normal);

  // Draw functions/operators
  void Draw(void) const;

  // Arithmetic functions/operators
  R3Plane operator-(void) const;
	
 private:
  R3Vector v;
  double d;
};



// Public variables 

extern const R3Plane R3null_plane;
extern const R3Plane R3posxz_plane;
extern const R3Plane R3posxy_plane;
extern const R3Plane R3posyz_plane;
extern const R3Plane R3negxz_plane;
extern const R3Plane R3negxy_plane;
extern const R3Plane R3negyz_plane;
#define R3xz_plane R3posxz_plane
#define R3xy_plane R3posxy_plane
#define R3yz_plane R3posyz_plane



// Inline functions 

inline double R3Plane::
operator[](int i) const
{
  assert ((i>=0) && (i<=3));
  return ((i == 3) ? d : v[i]);
}



inline double& R3Plane::
operator[](int i) 
{
  assert ((i>=0) && (i<=3));
  return ((i == 3) ? d : v[i]);
}



inline double R3Plane::
A (void) const
{
  return v[0];
}



inline double R3Plane::
B (void) const
{
  return v[1];
}



inline double R3Plane::
C (void) const
{
  return v[2];
}



inline double R3Plane::
D (void) const
{
  return d;
}



inline const R3Vector& R3Plane::
Normal (void) const
{
  return v;
}



inline bool R3Plane::
IsZero (void) const
{
  // Return whether plane has zero normal vector
  return v.IsZero();
}



inline bool R3Plane::
operator==(const R3Plane& plane) const
{
  // Return whether plane is equal
  return ((v == plane.v) && (d == plane.d));
}



inline bool R3Plane::
operator!=(const R3Plane& plane) const
{
  // Return whether plane is not equal
  return (!(*this == plane));
}



inline R3Plane R3Plane::
operator-(void) const
{
  // Return plane with flipped orientation
  return R3Plane(-v, -d);
}



inline void R3Plane::
Flip(void) 
{
  v = -v;
  d = -d;
}



inline void R3Plane::
Align(const R3Vector& normal) 
{
  // Align plane normal - keep same distance to origin
  v = normal;
}




