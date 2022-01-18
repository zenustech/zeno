// Include file for the R2 line class 




// Class definition 

class R2Line {
 public:
  // Constructor functions
  R2Line(void);
  R2Line(const R2Line& line);
  R2Line(double a, double b, double c);
  R2Line(double array[3]);
  R2Line(const R2Point& point, const R2Vector& vector, bool normalized = false);
  R2Line(const R2Point& point1, const R2Point& point2);
  R2Line(double x1, double y1, double x2, double y2);

  // Property functions/operators
  double A(void) const;
  double B(void) const;
  double C(void) const;
  R2Point AnyPoint(void) const;
  R2Point ClosestPoint(const R2Point& point) const;
  R2Vector Vector(void) const;
  R2Vector Normal(void) const;
  bool operator==(const R2Line& line) const;
  bool operator!=(const R2Line& line) const;

  // Manipulation functions/operators
  void Flip(void);
  void Mirror(const R2Line& line);
  void Translate(const R2Vector& vector);
  void Rotate(const R2Point& origin, double angle);
  void Reset(const R2Point& point, const R2Vector& vector, bool normalized = false);
  R2Line& operator=(const R2Line& line);

  // Arithmetic functions/operators
  R2Line operator-(void);
	
  // Output functions
  void Print(FILE *fp = stdout) const;

 private:
  R2Vector vector;
  R2Vector normal;
  double c;
};



// Public variables 

extern R2Line R2null_line;
extern R2Line R2posx_line;
extern R2Line R2posy_line;
extern R2Line R2negx_line;
extern R2Line R2negy_line;
#define R2xaxis_line R2posx_line
#define R2yaxis_line R2posy_line



// Inline functions 

inline double R2Line::
A(void) const
{
  // Return A coefficient of AX+BY+C=0
  return normal.X();
}



inline double R2Line::
B(void) const
{
  // Return B coefficient of AX+BY+C=0
  return normal.Y();
}



inline double R2Line::
C(void) const
{
  // Return C coefficient of AX+BY+C=0
  return c;
}



inline R2Point R2Line::
AnyPoint(void) const
{
  // Return point on line
  return R2zero_point + normal * -c;
}



inline R2Point R2Line::
ClosestPoint(const R2Point& point) const
{
  // Return point on line closest to given point
  double d = A()*point[0] + B()*point[1] + C();
  return point - d * normal;
}



inline R2Vector R2Line::
Vector(void) const
{
  // Return direction vector of line
  return vector;
}



inline R2Vector R2Line::
Normal(void) const
{
  // Return normal vector of line 
  // This vector is counterclockwise by 90 degrees with respect to direction vector
  return normal;
}



inline bool R2Line::
operator==(const R2Line& line) const
{
  // Return whether line is same
  if (normal != line.normal) return false;
  if (c != line.c) return false;
  return true;
}



inline bool R2Line::
operator!=(const R2Line& line) const
{
  // Return whether line is not equal
  return (!(*this == line));
}



