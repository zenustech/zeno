// Include file for the R3 ray class 



// Class definition 

class R3Ray {
 public:
  // Constructor functions
  R3Ray(void);
  R3Ray(const R3Ray& ray);
  R3Ray(const R3Point& point, const R3Vector& vector, bool normalized = false);
  R3Ray(const R3Point& point1, const R3Point& point2);
  R3Ray(double x1, double y1, double z1, double x2, double y2, double z2);

  // Property functions/operators
  const R3Point& Start(void) const;
  const R3Vector& Vector(void) const;
  const R3Line& Line(void) const;
  const R3Point Point(double t) const;
  const double T(const R3Point& point) const;
  const bool IsZero(void) const;
  const bool operator==(const R3Ray& ray) const;
  const bool operator!=(const R3Ray& ray) const;

  // Manipulation functions/operators
  void Flip(void);
  void Mirror(const R3Plane& plane);
  void Translate(const R3Vector& vector);
  void Reposition(const R3Point& point);
  void Align(const R3Vector& vector, bool normalized = false);
  void Transform(const R3Matrix& matrix);
  void InverseTransform(const R3Matrix& matrix);
  void Reset(const R3Point& point, const R3Vector& vector, bool normalized = false);

  // Arithmetic functions/operators
  R3Ray operator-(void) const;
	
  // Output functions
  void Draw(void) const;
  void Print(FILE *fp = stdout) const;

 private:
  R3Line line;
};



// Public variables 

extern const R3Ray R3null_ray;
extern const R3Ray R3posx_ray;
extern const R3Ray R3posy_ray;
extern const R3Ray R3posz_ray;
extern const R3Ray R3negx_ray;
extern const R3Ray R3negy_ray;
extern const R3Ray R3negz_ray;
#define R3xaxis_ray R3posx_ray
#define R3yaxis_ray R3posy_ray
#define R3zaxis_ray R3posz_ray



// Inline functions 

inline const R3Point& R3Ray::
Start(void) const
{
  // Return source point of ray
  return line.Point();
}



inline const R3Vector& R3Ray::
Vector(void) const
{
  // Return direction vector of ray
  return line.Vector();
}



inline const R3Line& R3Ray::
Line(void) const
{
  // Return line containing ray
  return line;
}



inline const bool R3Ray::
IsZero (void) const
{
  // Return whether ray has zero vector
  return line.IsZero();
}



inline const bool R3Ray::
operator==(const R3Ray& ray) const
{
  // Return whether ray is equal
  return (line == ray.line);
}



inline const bool R3Ray::
operator!=(const R3Ray& ray) const
{
  // Return whether ray is not equal
  return (!(*this == ray));
}



inline R3Ray R3Ray::
operator-(void) const
{
  // Return ray with flipped orientation
  return R3Ray(line.Point(), -(line.Vector()));
}



inline void R3Ray::
Flip(void)
{
  // Flip direction of ray
  line.Flip();
}



inline void R3Ray::
Mirror(const R3Plane& plane)
{
  // Mirror ray over plane
  line.Mirror(plane);
}



inline void R3Ray::
Translate(const R3Vector& vector)
{
  // Move endpoint of ray
  line.Translate(vector);
}



inline void R3Ray::
Reposition(const R3Point& point)
{
  // Set endpoint of ray
  line.Reposition(point);
}



inline void R3Ray::
Align(const R3Vector& vector, bool normalized)
{
  // Set vector of ray
  line.Align(vector, normalized);
}



inline void R3Ray::
Reset(const R3Point& point, const R3Vector& vector, bool normalized)
{
  // Reset ray
  line.Reset(point, vector, normalized);
}



