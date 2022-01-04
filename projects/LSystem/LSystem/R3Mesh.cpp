// Source file for mesh class



// Include files
#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include "R3Mesh.h"
#include "lplus.h"

void R3Mesh::
Twist(double angle)
{
  // Twist mesh by an angle, or other simple mesh warping of your choice.
  // See Scale() for how to get the vertex positions, and see bbox for the bounding box.

  // FILL IN IMPLEMENTATION HERE

  // Update mesh data structures
  Update();
}
R3Shape R3Mesh::Leaf(const R3Vector direction)
{
  float z;
  z=direction.Dot(R3Vector(0,1,0))/4.0; //bend towards earth
  
  if (z==0) z=(rand()%20 -10 ) /100.0; //some random bend if non

  vector<R3MeshVertex *> face;
  face.push_back(CreateVertex(R3Point(0,.01,0)  ,R2Point(.5,.01) )); 
  face.push_back(CreateVertex(R3Point(.2,.1,0)  ,R2Point(.7,.1) ));
  face.push_back(CreateVertex(R3Point(.25,.3,0) ,R2Point(.75,.3) ));
  face.push_back(CreateVertex(R3Point(.2,.6,z/2) ,R2Point(.7,.6) ));
  

  face.push_back(CreateVertex(R3Point(0,1-z,z) ,R2Point(.5,1) ));
  face.push_back(CreateVertex(R3Point(-.2,.6,z/2) ,R2Point(.3,.6) ));
  face.push_back(CreateVertex(R3Point(-.25,.3,0) ,R2Point(.25,.3) ));
  face.push_back(CreateVertex(R3Point(-.2,.1,0) ,R2Point(.3,.1) ));
  CreateFace(face)->isLeaf=true;
  return face;

}
R3Shape R3Mesh::Circle(float radius,int slices)
{
  vector<R3MeshVertex *> face_vertices;
  for(int i=0; i<slices; i++) 
  {
    R3MeshVertex*t;
    float theta = ((float)i)* (2.0*M_PI/slices);
    t=CreateVertex(R3Point(radius*cos(theta), 0, radius*sin(theta))); //vertices at edges of circle
    face_vertices.push_back(t);
  }
  CreateFace(face_vertices);
  return face_vertices;

}
R3Shape R3Mesh::Cylinder(float topBottomRatio,int slices)
{
  static bool cached=false;
  cached=false;
  static vector<R3Point> cache;
  int cacheIndex=0;

  float length=1,radius=1;
  float topRadius=topBottomRatio;
  R3Shape vertices;
  R3Shape bottom_circle;
  R3Shape top_circle;
  R3Shape side;
  for(int i=0; i<slices; i++) 
  {
    R3MeshVertex*t1,*t2;
    R3Point p;
    float theta = ((float)i)* (2.0*M_PI/slices);

    if (!cached) cache.push_back( p=R3Point(topRadius*cos(theta), length, topRadius*sin(theta)) );
    else p=cache[cacheIndex++];
    t1=CreateVertex(p,R2Point(i*2/(float)slices,1)) ; //vertices at edges of circle
    top_circle.push_back(t1);
    vertices.push_back(t1);

    if (!cached) cache.push_back( p=R3Point(radius*cos(theta), 0, radius*sin(theta)) );
    else p=cache[cacheIndex++];
    t2=CreateVertex(p,R2Point(i*2/(float)slices,0)); //vertices at edges of circle
    bottom_circle.push_back(t2);
    vertices.push_back(t2);
  }
  int size=vertices.size();
  for (int i=0;i<size;i+=2)
  {

    side.push_back(vertices[i]);
    side.push_back(vertices[i+1]);
    side.push_back(vertices[(i+2)%size]);
    CreateFace(side);
    side.clear();

    side.push_back(vertices[i+1]);
    side.push_back(vertices[(i+3)%size]);
    side.push_back(vertices[(i+2)%size]);
    CreateFace(side);
    side.clear();
  }
  CreateFace(top_circle);
  CreateFace(bottom_circle);
  if (!cached)
    cached=true;
  return vertices;
}
void R3Mesh::AddCoords()
{
  float width=.05;
  R3Shape shape;
  shape.push_back(CreateVertex(R3Point(3,0,0)));
  shape.push_back(CreateVertex(R3Point(0,width,0)));
  shape.push_back(CreateVertex(R3Point(0,-width,0)));
  CreateFace(shape);
  shape.clear();
  shape.push_back(CreateVertex(R3Point(0,2,0)));
  shape.push_back(CreateVertex(R3Point(-width,0,0)));
  shape.push_back(CreateVertex(R3Point(width,0,0)));
  CreateFace(shape);
  shape.clear();
  shape.push_back(CreateVertex(R3Point(0,0,1)));
  shape.push_back(CreateVertex(R3Point(0,-width,0)));
  shape.push_back(CreateVertex(R3Point(0,width,0)));
  CreateFace(shape);


}
void R3Mesh::
Tree(const string code, const bool isPlus)
{
  /** turtle system test *
    TurtleSystem t(this);
  t.pitchDown(90);
  t.turnRight(90);
  t.move(10);
  printf("%f %f %f\n",t.position.X(),t.position.Y(),t.position.Z());
  printf("%f %f %f\n",t.direction.X(),t.direction.Y(),t.direction.Z());
  printf("%f %f %f\n",t.right.X(),t.right.Y(),t.right.Z());
  * end turtle system test **/


  // AddCoords();
  // R3Shape cylinder=Cylinder();      prim->tris[i]=ze
  // Update();
  // return;

  // Leaf(R3Vector(0,1,0));
  // Update();
  // return;

  LPlusSystem l(this);
  string lsystem=l.generateFromCode(code, isPlus);
  l.draw(lsystem); 
  Update();

}
////////////////////////////////////////////////////////////
// MESH CONSTRUCTORS/DESTRUCTORS
////////////////////////////////////////////////////////////

R3Mesh::
R3Mesh(void)
: bbox(R3null_box)
{
}



R3Mesh::
R3Mesh(const R3Mesh& mesh)
: bbox(R3null_box)
{
  // Create vertices
  for (int i = 0; i < mesh.NVertices(); i++) {
    R3MeshVertex *v = mesh.Vertex(i);
    CreateVertex(v->position, v->normal, v->texcoords);
  }

  // Create faces
  for (int i = 0; i < mesh.NFaces(); i++) {
    R3MeshFace *f = mesh.Face(i);
    vector<R3MeshVertex *> face_vertices;
    for (unsigned int j = 0; j < f->vertices.size(); j++) {
      R3MeshVertex *ov = f->vertices[j];
      R3MeshVertex *nv = Vertex(ov->id);
      face_vertices.push_back(nv);
    }
    CreateFace(face_vertices);
  }
}



R3Mesh::
~R3Mesh(void)
{
  // Delete faces
  for (int i = 0; i < NFaces(); i++) {
    R3MeshFace *f = Face(i);
    delete f;
  }

  // Delete vertices
  for (int i = 0; i < NVertices(); i++) {
    R3MeshVertex *v = Vertex(i);
    delete v;
  }
}



////////////////////////////////////////////////////////////
// MESH PROPERTY FUNCTIONS
////////////////////////////////////////////////////////////

R3Point R3Mesh::
Center(void) const
{
  // Return center of bounding box
  return bbox.Centroid();
}



double R3Mesh::
Radius(void) const
{
  // Return radius of bounding box
  return bbox.DiagonalRadius();
}



////////////////////////////////////////////////////////////
// MESH PROCESSING FUNCTIONS
////////////////////////////////////////////////////////////

void R3Mesh::TranslateShape(vector<R3MeshVertex *> shape,double dx,double dy,double dz)
{
  R3Vector translation(dx, dy, dz);

  // Update vertices
  for (unsigned int i = 0; i < shape.size(); i++) {
    R3MeshVertex *vertex = shape[i];
    vertex->position.Translate(translation);
  }

  // Update mesh data structures
}
void R3Mesh::
Translate(double dx, double dy, double dz)
{
  TranslateShape(vertices,dx,dy,dz);
  Update();
}




void R3Mesh::
ScaleShape(vector<R3MeshVertex *> shape,double sx, double sy, double sz)
{
  // Scale the mesh by increasing the distance 
  // from every vertex to the origin by a factor 
  // given for each dimension (sx, sy, sz)

  // This is implemented for you as an example 

  // Update vertices
  for (unsigned int i = 0; i < shape.size(); i++) {
    R3MeshVertex *vertex = shape[i];
    vertex->position[0] *= sx;
    vertex->position[1] *= sy;
    vertex->position[2] *= sz;
  }

  // Update mesh data structures
}
void R3Mesh::Scale(double sx,double sy,double sz)
{
  ScaleShape(vertices,sx,sy,sz);
  Update();
}

void R3Mesh::
RotateShape(vector<R3MeshVertex *> shape,double angle, const R3Vector& axis)
{
  for (unsigned int i = 0; i < shape.size(); i++) {
    R3MeshVertex *vertex = shape[i];
    vertex->position.Rotate(axis, angle);
  }

  // Update mesh data structures

}

void R3Mesh::
RotateShape(vector<R3MeshVertex *> shape,double angle, const R3Line& axis)
{
  // Rotate the mesh counter-clockwise by an angle 
  // (in radians) around a line axis

  // This is implemented for you as an example 

  // Update vertices
  for (unsigned int i = 0; i < shape.size(); i++) {
    R3MeshVertex *vertex = shape[i];
    vertex->position.Rotate(axis, angle);
  }

  // Update mesh data structures
}
void R3Mesh::Rotate(double angle, const R3Line& axis)
{
  RotateShape(vertices,angle,axis);
  Update();
}


////////////////////////////////////////////////////////////
// MESH ELEMENT CREATION/DELETION FUNCTIONS
////////////////////////////////////////////////////////////
R3MeshVertex *R3Mesh::CreateVertex(const R3Point& position, const R2Point& texcoords)
{
  return CreateVertex(position,R3zero_vector,texcoords);
}

R3MeshVertex *R3Mesh::CreateVertex(const R3Point& position, const R3Vector& normal, const R2Point& texcoords)
{
  R2Point tx=texcoords;
  // Create vertex
  R3MeshVertex *vertex = new R3MeshVertex(position, normal, tx);

  // Update bounding box
  bbox.Union(position);

  // Set vertex ID
  vertex->id = vertices.size();

  // Add to list
  vertices.push_back(vertex);

  // Return vertex
  return vertex;
}



R3MeshFace *R3Mesh::
CreateFace(const vector<R3MeshVertex *>& vertices)
{
  // Create face
  R3MeshFace *face = new R3MeshFace(vertices);

  // Set face  ID
  face->id = faces.size();

  // Add to list
  faces.push_back(face);

  // Return face
  return face;
}



void R3Mesh::
DeleteVertex(R3MeshVertex *vertex)
{
  // Remove vertex from list
  for (unsigned int i = 0; i < vertices.size(); i++) {
    if (vertices[i] == vertex) {
      vertices[i] = vertices.back();
      vertices[i]->id = i;
      vertices.pop_back();
      break;
    }
  }

  // Delete vertex
  delete vertex;
}



void R3Mesh::
DeleteFace(R3MeshFace *face)
{
  // Remove face from list
  for (unsigned int i = 0; i < faces.size(); i++) {
    if (faces[i] == face) {
      faces[i] = faces.back();
      faces[i]->id = i;
      faces.pop_back();
      break;
    }
  }

  // Delete face
  delete face;
}



////////////////////////////////////////////////////////////
// UPDATE FUNCTIONS
////////////////////////////////////////////////////////////

void R3Mesh::
Update(void)
{
  // Update everything
  UpdateBBox();
  UpdateFacePlanes();
  UpdateVertexNormals();
  UpdateVertexCurvatures();
}



void R3Mesh::
UpdateBBox(void)
{
  // Update bounding box
  bbox = R3null_box;
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3MeshVertex *vertex = vertices[i];
    bbox.Union(vertex->position);
  }
}



void R3Mesh::
UpdateVertexNormals(void)
{
  // Update normal for every vertex
  for (unsigned int i = 0; i < vertices.size(); i++) {
    vertices[i]->UpdateNormal();
  }
}




void R3Mesh::
UpdateVertexCurvatures(void)
{
  // Update curvature for every vertex
  for (unsigned int i = 0; i < vertices.size(); i++) {
    vertices[i]->UpdateCurvature();
  }
}




void R3Mesh::
UpdateFacePlanes(void)
{
  // Update plane for all faces
  for (unsigned int i = 0; i < faces.size(); i++) {
    faces[i]->UpdatePlane();
  }
}









////////////////////////////////////////////////////////////
// MESH VERTEX MEMBER FUNCTIONS
////////////////////////////////////////////////////////////

R3MeshVertex::
R3MeshVertex(void)
: position(0, 0, 0),
normal(0, 0, 0),
texcoords(0, 0),
curvature(0),
id(0)
{
}



R3MeshVertex::
R3MeshVertex(const R3MeshVertex& vertex)
: position(vertex.position),
normal(vertex.normal),
texcoords(vertex.texcoords),
curvature(vertex.curvature),
id(0)
{
}




R3MeshVertex::
R3MeshVertex(const R3Point& position, const R3Vector& normal, const R2Point& texcoords)
: position(position),                    
normal(normal),
texcoords(texcoords),
curvature(0),
id(0)
{
}




double R3MeshVertex::
AverageEdgeLength(void) const
{
  // Return the average length of edges attached to this vertex
  // This feature should be implemented first.  To do it, you must
  // design a data structure that allows O(K) access to edges attached
  // to each vertex, where K is the number of edges attached to the vertex.

  // FILL IN IMPLEMENTATION HERE  (THIS IS REQUIRED)
  // BY REPLACING THIS ARBITRARY RETURN VALUE
  fprintf(stderr, "Average vertex edge length not implemented\n");
  return 0.12345;
}




void R3MeshVertex::
UpdateNormal(void)
{
  // Compute the surface normal at a vertex.  This feature should be implemented
  // second.  To do it, you must design a data structure that allows O(K)
  // access to faces attached to each vertex, where K is the number of faces attached
  // to the vertex.  Then, to compute the normal for a vertex,
  // you should take a weighted average of the normals for the attached faces, 
  // where the weights are determined by the areas of the faces.
  // Store the resulting normal in the "normal"  variable associated with the vertex. 
  // You can display the computed normals by hitting the 'N' key in meshview.

  // FILL IN IMPLEMENTATION HERE (THIS IS REQUIRED)
  // fprintf(stderr, "Update vertex normal not implemented\n");
}




void R3MeshVertex::
UpdateCurvature(void)
{
  // Compute an estimate of the Gauss curvature of the surface 
  // using a method based on the Gauss Bonet Theorem, which is described in 
  // [Akleman, 2006]. Store the result in the "curvature"  variable. 

  // FILL IN IMPLEMENTATION HERE
  // fprintf(stderr, "Update vertex curvature not implemented\n");
}





////////////////////////////////////////////////////////////
// MESH FACE MEMBER FUNCTIONS
////////////////////////////////////////////////////////////

R3MeshFace::
R3MeshFace(void)
: vertices(),
plane(0, 0, 0, 0),
id(0),
isLeaf(0)
{
}



R3MeshFace::
R3MeshFace(const R3MeshFace& face)
: vertices(face.vertices),
plane(face.plane),
id(0),
isLeaf(0)
{
}



R3MeshFace::
R3MeshFace(const vector<R3MeshVertex *>& vertices)
: vertices(vertices),
plane(0, 0, 0, 0),
id(0),
isLeaf(0)
{
  UpdatePlane();
}



double R3MeshFace::
AverageEdgeLength(void) const
{
  // Check number of vertices
  if (vertices.size() < 2) return 0;

  // Compute average edge length
  double sum = 0;
  R3Point *p1 = &(vertices.back()->position);
  for (unsigned int i = 0; i < vertices.size(); i++) {
    R3Point *p2 = &(vertices[i]->position);
    double edge_length = R3Distance(*p1, *p2);
    sum += edge_length;
    p1 = p2;
  }

  // Return the average length of edges attached to this face
  return sum / vertices.size();
}



double R3MeshFace::
Area(void) const
{
  // Check number of vertices
  if (vertices.size() < 3) return 0;

  // Compute area using Newell's method (assumes convex polygon)
  R3Vector sum = R3null_vector;
  const R3Point *p1 = &(vertices.back()->position);
  for (unsigned int i = 0; i < vertices.size(); i++) {
    const R3Point *p2 = &(vertices[i]->position);
    sum += p2->Vector() % p1->Vector();
    p1 = p2;
  }

  // Return area
  return 0.5 * sum.Length();
}



void R3MeshFace::
UpdatePlane(void)
{
  // Check number of vertices
  int nvertices = vertices.size();
  if (nvertices < 3) { 
    plane = R3null_plane; 
    return; 
  }

  // Compute centroid
  R3Point centroid = R3zero_point;
  for (int i = 0; i < nvertices; i++) 
    centroid += vertices[i]->position;
  centroid /= nvertices;
  
  // Compute best normal for counter-clockwise array of vertices using newell's method
  R3Vector normal = R3zero_vector;
  const R3Point *p1 = &(vertices[nvertices-1]->position);
  for (int i = 0; i < nvertices; i++) {
    const R3Point *p2 = &(vertices[i]->position);
    normal[0] += (p1->Y() - p2->Y()) * (p1->Z() + p2->Z());
    normal[1] += (p1->Z() - p2->Z()) * (p1->X() + p2->X());
    normal[2] += (p1->X() - p2->X()) * (p1->Y() + p2->Y());
    p1 = p2;
  }
  
  // Normalize normal vector
  normal.Normalize();
  
  // Update face plane
  plane.Reset(centroid, normal);
}



