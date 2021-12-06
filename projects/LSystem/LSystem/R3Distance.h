// Include file for R3 distance utility 



// Function declarations 

double R3Distance(const R3Point& point1, const R3Point& point2);
double R3Distance(const R3Point& point, const R3Line& line);
double R3Distance(const R3Point& point, const R3Ray& ray);
double R3Distance(const R3Point& point, const R3Segment& segment);
double R3Distance(const R3Point& point, const R3Plane& plane);
double R3Distance(const R3Point& point, const R3Box& box);
double R3SquaredDistance(const R3Point& point1, const R3Point& point2);

double R3Distance(const R3Line& line, const R3Point& point);
double R3Distance(const R3Line& line1, const R3Line& line2);
double R3Distance(const R3Line& line, const R3Ray& ray);
double R3Distance(const R3Line& line, const R3Segment& segment);
double R3Distance(const R3Line& line, const R3Plane& plane);
double R3Distance(const R3Line& line, const R3Box& box);

double R3Distance(const R3Ray& ray, const R3Point& point);
double R3Distance(const R3Ray& ray, const R3Line& line);
double R3Distance(const R3Ray& ray1, const R3Ray& ray2);
double R3Distance(const R3Ray& ray, const R3Segment& segment);
double R3Distance(const R3Ray& ray, const R3Plane& plane);
double R3Distance(const R3Ray& ray, const R3Box& box);

double R3Distance(const R3Segment& segment, const R3Point& point);
double R3Distance(const R3Segment& segment, const R3Line& line);
double R3Distance(const R3Segment& segment, const R3Ray& ray);
double R3Distance(const R3Segment& segment1, const R3Segment& segment2);
double R3Distance(const R3Segment& segment, const R3Plane& plane);
double R3Distance(const R3Segment& segment, const R3Box& box);

double R3Distance(const R3Plane& plane, const R3Point& point);
double R3Distance(const R3Plane& plane, const R3Line& line);
double R3Distance(const R3Plane& plane, const R3Ray& ray);
double R3Distance(const R3Plane& plane, const R3Segment& segment);
double R3Distance(const R3Plane& plane1, const R3Plane& plane2);
double R3Distance(const R3Plane& plane, const R3Box& box);

double R3SignedDistance(const R3Plane& plane, const R3Point& point);
double R3SignedDistance(const R3Plane& plane, const R3Line& line);
double R3SignedDistance(const R3Plane& plane, const R3Ray& ray);
double R3SignedDistance(const R3Plane& plane, const R3Segment& segment);
double R3SignedDistance(const R3Plane& plane1, const R3Plane& plane2);
double R3SignedDistance(const R3Plane& plane, const R3Box& box);

double R3Distance(const R3Box& box, const R3Point& point);
double R3Distance(const R3Box& box, const R3Line& line);
double R3Distance(const R3Box& box, const R3Ray& ray);
double R3Distance(const R3Box& box, const R3Segment& segment);
double R3Distance(const R3Box& box, const R3Plane& plane);
double R3Distance(const R3Box& box1, const R3Box& box2);



// Inline functions 

inline double R3Distance(const R3Line& line, const R3Point& point)
{
    // Distance is commutative
    return R3Distance(point, line);
}



inline double R3Distance(const R3Ray& ray, const R3Point& point)
{
    // Distance is commutative
    return R3Distance(point, ray);
}



inline double R3Distance(const R3Ray& ray, const R3Line& line)
{
    // Distance is commutative
    return R3Distance(line, ray);
}



inline double R3Distance(const R3Segment& segment, const R3Point& point)
{
    // Distance is commutative
    return R3Distance(point, segment);
}



inline double R3Distance(const R3Segment& segment, const R3Line& line)
{
    // Distance is commutative
    return R3Distance(line, segment);
}



inline double R3Distance(const R3Segment& segment, const R3Ray& ray)
{
    // Distance is commutative
    return R3Distance(ray, segment);
}



inline double R3Distance(const R3Plane& plane, const R3Point& point)
{
    // Distance is commutative
    return R3Distance(point, plane);
}



inline double R3Distance(const R3Plane& plane, const R3Line& line)
{
    // Distance is commutative
    return R3Distance(line, plane);
}



inline double R3Distance(const R3Plane& plane, const R3Ray& ray)
{
    // Distance is commutative
    return R3Distance(ray, plane);
}



inline double R3Distance(const R3Plane& plane, const R3Segment& segment)
{
    // Distance is commutative
    return R3Distance(segment, plane);
}



inline double R3Distance(const R3Box& box, const R3Point& point)
{
    // Distance is commutative
    return R3Distance(point, box);
}



inline double R3Distance(const R3Box& box, const R3Line& line)
{
    // Distance is commutative
    return R3Distance(line, box);
}



inline double R3Distance(const R3Box& box, const R3Ray& ray)
{
    // Distance is commutative
    return R3Distance(ray, box);
}



inline double R3Distance(const R3Box& box, const R3Segment& segment)
{
    // Distance is commutative
    return R3Distance(segment, box);
}



inline double R3Distance(const R3Box& box, const R3Plane& plane)
{
    // Distance is commutative
    return R3Distance(plane, box);
}



