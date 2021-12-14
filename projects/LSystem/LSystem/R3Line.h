// Include file for the R3 line class 



// Class definition 

class R3Line {
 public:
  // Constructor functions
  R3Line(void);
  R3Line(const R3Line& line);
  R3Line(const R3Point& point, const R3Vector& vector, bool normalized = false);
  R3Line(const R3Point& point1, const R3Point& point2);
  R3Line(double x1, double y1, double z1, double x2, double y2, double z2);

  // Property functions/operators
  const R3Point& Point(void) const;
  const R3Vector& Vector(void) const;
  bool IsZero(void) const;
  bool operator==(const R3Line& line) const;
  bool operator!=(const R3Line& line) const;

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
  R3Line operator-(void) const;
	
  // Draw functions/operators
  void Draw(void) const;
  void Print(FILE *fp = stdout) const;

 private:
  R3Point point;
  R3Vector vector;
};



// Public variables 

extern const R3Line R3null_line;
extern const R3Line R3posx_line;
extern const R3Line R3posy_line;
extern const R3Line R3posz_line;
extern const R3Line R3negx_line;
extern const R3Line R3negy_line;
extern const R3Line R3negz_line;
#define R3xaxis_line R3posx_line
#define R3yaxis_line R3posy_line
#define R3zaxis_line R3posz_line



// Inline functions 

inline const R3Point& R3Line::
Point(void) const
{
  // Return point on line
  return point;
}



inline const R3Vector& R3Line::
Vector(void) const
{
  // Return direction vector of line
  return vector;
}



inline bool R3Line::
IsZero (void) const
{
  // Return whether line has zero vector
  return vector.IsZero();
}



inline bool R3Line::
operator!=(const R3Line& line) const
{
  // Return whether line is not equal
  return (!(*this == line));
}



inline R3Line R3Line::
operator-(void) const
{
  // Return line with flipped orientation
  return R3Line(point, -vector);
}



inline void R3Line::
Flip(void)
{
  // Flip direction of line
  vector.Flip();
}



inline void R3Line::
Mirror(const R3Plane& plane)
{
  // Mirror line over plane
  point.Mirror(plane);
  vector.Mirror(plane);
}



inline void R3Line::
Translate(const R3Vector& vector)
{
  // Move point on line
  this->point += vector;
}



inline void R3Line::
Reposition(const R3Point& point)
{
  // Set point on line
  this->point = point;
}



inline void R3Line::
Align(const R3Vector& vector, bool normalized)
{
  // Set vector of line
  this->vector = vector;
  if (!normalized) this->vector.Normalize();
}



inline void R3Line::
Reset(const R3Point& point, const R3Vector& vector, bool normalized)
{
  // Reset line
  this->point = point;
  this->vector = vector;
  if (!normalized) this->vector.Normalize();
}



