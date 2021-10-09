#include "meshfix.h"
#include <cstring>
#include <cstdlib>
#include <jrs_predicates.h>

//#define MESHFIX_VERBOSE

// Simulates the ASCII rounding error
void asciiAlign(ExtTriMesh& tin)
{
 char outname[2048];
 Vertex *v;
 Node *m;
 float a;
 FOREACHVVVERTEX((&(tin.V)), v, m)
 {
  sprintf(outname,"%f",v->x); sscanf(outname,"%f",&a); v->x = a;
  sprintf(outname,"%f",v->y); sscanf(outname,"%f",&a); v->y = a;
  sprintf(outname,"%f",v->z); sscanf(outname,"%f",&a); v->z = a;
 }
}


// Return TRUE if the triangle is exactly degenerate

inline bool isDegenerateEdge(Edge *e)
{
 return ((*(e->v1))==(*(e->v2)));
}

bool isDegenerateTriangle(Triangle *t)
{
 double xy1[2], xy2[2], xy3[2];
 xy1[0] = t->v1()->x; xy1[1] = t->v1()->y; 
 xy2[0] = t->v2()->x; xy2[1] = t->v2()->y; 
 xy3[0] = t->v3()->x; xy3[1] = t->v3()->y; 
 if (orient2d(xy1, xy2, xy3)!=0.0) return false;
 xy1[0] = t->v1()->y; xy1[1] = t->v1()->z; 
 xy2[0] = t->v2()->y; xy2[1] = t->v2()->z; 
 xy3[0] = t->v3()->y; xy3[1] = t->v3()->z; 
 if (orient2d(xy1, xy2, xy3)!=0.0) return false;
 xy1[0] = t->v1()->z; xy1[1] = t->v1()->x; 
 xy2[0] = t->v2()->z; xy2[1] = t->v2()->x; 
 xy3[0] = t->v3()->z; xy3[1] = t->v3()->x; 
 if (orient2d(xy1, xy2, xy3)!=0.0) return false;
 return true;
}


Edge *getLongestEdge(Triangle *t)
{
 double l1 = t->e1->squaredLength();
 double l2 = t->e2->squaredLength();
 double l3 = t->e3->squaredLength();
 if (l1>=l2 && l1>=l3) return t->e1;
 if (l2>=l1 && l2>=l3) return t->e2;
 return t->e3;
}


// Iterate on all the selected triangles as long as possible.
// Keep the selection only on the degeneracies that could not be removed.
// Return the number of degeneracies that could not be removed
int swap_and_collapse(const double epsilon_angle, ExtTriMesh *tin)
{
 Node *n;
 Triangle *t;

 if (epsilon_angle != 0.0)
 {
  FOREACHVTTRIANGLE((&(tin->T)), t, n) UNMARK_VISIT(t);
 
#ifdef MESHFIX_VERBOSE
  JMesh::quiet = true;
#endif
 tin->removeDegenerateTriangles();
#ifdef MESHFIX_VERBOSE
  JMesh::quiet = false;
#endif
 
  int failed = 0;
  FOREACHVTTRIANGLE((&(tin->T)), t, n) if (IS_VISITED(t)) failed++;
  return failed;
 }

 List triangles;
 Edge *e;
 const int MAX_ATTEMPTS = 10;

 FOREACHVTTRIANGLE((&(tin->T)), t, n) t->info=0;

 // VISIT2 means that the triangle is in the list
 FOREACHVTTRIANGLE((&(tin->T)), t, n) if (IS_VISITED(t))
 {
  UNMARK_VISIT(t);
  if (isDegenerateTriangle(t)) {triangles.appendTail(t); MARK_VISIT2(t);}
 }

 while ((t=(Triangle *)triangles.popHead())!=NULL)
 {
  UNMARK_VISIT2(t);
  if (t->isLinked())
  {
   if (isDegenerateEdge(t->e1)) t->e1->collapse();
   else if (isDegenerateEdge(t->e2)) t->e2->collapse();
   else if (isDegenerateEdge(t->e3)) t->e3->collapse();
   else if ((e=getLongestEdge(t))!=NULL)
   {
    if (e->swap())
	{
	 t=e->t1;
           // Alec: replaced "int" with "j_voidint"
	 if (isDegenerateTriangle(t) && !IS_VISITED2(t) && ((j_voidint)t->info < MAX_ATTEMPTS))
	  {triangles.appendTail(t); MARK_VISIT2(t); t->info = (void *)(((j_voidint)t->info)+1);}
	 t=e->t2;
	 if (isDegenerateTriangle(t) && !IS_VISITED2(t) && ((j_voidint)t->info < MAX_ATTEMPTS))
	  {triangles.appendTail(t); MARK_VISIT2(t); t->info = (void *)(((j_voidint)t->info)+1);}
	}
   }
  }
 }

 tin->removeUnlinkedElements();

 int failed=0;
 // This should check only on actually processed triangles
 FOREACHVTTRIANGLE((&(tin->T)), t, n) if (isDegenerateTriangle(t)) {failed++; MARK_VISIT(t);}

 JMesh::info("%d degeneracies selected\n",failed);
 return failed;
}

// returns true on success

bool removeDegenerateTriangles(const double epsilon_angle, ExtTriMesh& tin, int max_iters)
{
 int n, iter_count = 0;

#ifdef MESHFIX_VERBOSE
 printf("Removing degeneracies...\n");
#endif
 while ((++iter_count) <= max_iters && swap_and_collapse(epsilon_angle, &tin))
 {
  for (n=1; n<iter_count; n++) tin.growSelection();
  tin.removeSelectedTriangles();
  tin.removeSmallestComponents();
 
#ifdef MESHFIX_VERBOSE
  JMesh::quiet = true;
#endif
 tin.fillSmallBoundaries(tin.E.numels());
#ifdef MESHFIX_VERBOSE
  JMesh::quiet = false;
#endif

  asciiAlign(tin);
 }

 if (iter_count > max_iters) return false;
 return true;
}









bool appendCubeToList(Triangle *t0, List& l)
{
 if (!IS_VISITED(t0) || IS_VISITED2(t0)) return false;

 Triangle *t, *s;
 Vertex *v;
 List triList(t0);
 MARK_VISIT2(t0);
 double minx=DBL_MAX, maxx=-DBL_MAX, miny=DBL_MAX, maxy=-DBL_MAX, minz=DBL_MAX, maxz=-DBL_MAX;

 while(triList.numels())
 {
  t = (Triangle *)triList.popHead();
  v = t->v1();
  minx=MIN(minx,v->x); miny=MIN(miny,v->y); minz=MIN(minz,v->z);
  maxx=MAX(maxx,v->x); maxy=MAX(maxy,v->y); maxz=MAX(maxz,v->z);
  v = t->v2();
  minx=MIN(minx,v->x); miny=MIN(miny,v->y); minz=MIN(minz,v->z);
  maxx=MAX(maxx,v->x); maxy=MAX(maxy,v->y); maxz=MAX(maxz,v->z);
  v = t->v3();
  minx=MIN(minx,v->x); miny=MIN(miny,v->y); minz=MIN(minz,v->z);
  maxx=MAX(maxx,v->x); maxy=MAX(maxy,v->y); maxz=MAX(maxz,v->z);
  if ((s = t->t1()) != NULL && !IS_VISITED2(s) && IS_VISITED(s)) {triList.appendHead(s); MARK_VISIT2(s);}
  if ((s = t->t2()) != NULL && !IS_VISITED2(s) && IS_VISITED(s)) {triList.appendHead(s); MARK_VISIT2(s);}
  if ((s = t->t3()) != NULL && !IS_VISITED2(s) && IS_VISITED(s)) {triList.appendHead(s); MARK_VISIT2(s);}
 }

 l.appendTail(new Point(minx, miny, minz));
 l.appendTail(new Point(maxx, maxy, maxz));
 return true;
}

bool isVertexInCube(Vertex *v, List& loc)
{
 Node *n;
 Point *p1, *p2;
 FOREACHNODE(loc, n)
 {
  p1 = (Point *)n->data; n=n->next(); p2 = (Point *)n->data;
  if (!(v->x < p1->x || v->y < p1->y || v->z < p1->z ||
      v->x > p2->x || v->y > p2->y || v->z > p2->z)) return true;
 }

 return false;
}

void selectTrianglesInCubes(ExtTriMesh& tin)
{
 Triangle *t;
 Vertex *v;
 Node *n;
 List loc;
 FOREACHVTTRIANGLE((&(tin.T)), t, n) appendCubeToList(t, loc);
 FOREACHVVVERTEX((&(tin.V)), v, n) if (isVertexInCube(v, loc)) MARK_VISIT(v);
 FOREACHVTTRIANGLE((&(tin.T)), t, n)
 {
  UNMARK_VISIT2(t);
  if (IS_VISITED(t->v1()) || IS_VISITED(t->v2()) || IS_VISITED(t->v3())) MARK_VISIT(t);
 }
 FOREACHVVVERTEX((&(tin.V)), v, n) UNMARK_VISIT(v);
 loc.freeNodes();
}







// returns true on success

bool removeSelfIntersections(ExtTriMesh& tin, int max_iters)
{
 int n, iter_count = 0;

#ifdef MESHFIX_VERBOSE
 printf("Removing self-intersections...\n");
#endif
 while ((++iter_count) <= max_iters && tin.selectIntersectingTriangles())
 {
  for (n=1; n<iter_count; n++) tin.growSelection();
  tin.removeSelectedTriangles();
  tin.removeSmallestComponents();
 
#ifdef MESHFIX_VERBOSE
  JMesh::quiet = true;
#endif
 tin.fillSmallBoundaries(tin.E.numels());
#ifdef MESHFIX_VERBOSE
  JMesh::quiet = false;
#endif

  asciiAlign(tin);
  selectTrianglesInCubes(tin);
 }

 if (iter_count > max_iters) return false;
 return true;
}


bool isDegeneracyFree(const double epsilon_angle, ExtTriMesh& tin)
{
 Node *n;
 Triangle *t;

 if (epsilon_angle != 0.0)
 {FOREACHVTTRIANGLE((&(tin.T)), t, n) if (t->isDegenerate()) return false;}
 else
 {FOREACHVTTRIANGLE((&(tin.T)), t, n) if (isDegenerateTriangle(t)) return false;}

 return true;
}


// returns true on success

bool meshclean(const double epsilon_angle, ExtTriMesh& tin, int max_iters = 10, int inner_loops = 3)
{
 bool ni, nd;

 tin.deselectTriangles();
 tin.invertSelection();

 for (int n=0; n<max_iters; n++)
 {
#ifdef MESHFIX_VERBOSE
  printf("********* ITERATION %d *********\n",n);
#endif
  nd=removeDegenerateTriangles(epsilon_angle,tin, inner_loops);
  tin.deselectTriangles(); tin.invertSelection();
  ni=removeSelfIntersections(tin, inner_loops);
  if (ni && nd && isDegeneracyFree(epsilon_angle,tin)) return true;
 }

 return false;
}



double closestPair(List *bl1, List *bl2, Vertex **closest_on_bl1, Vertex **closest_on_bl2)
{
 Node *n, *m;
 Vertex *v,*w;
 double adist, mindist = DBL_MAX;

 FOREACHVVVERTEX(bl1, v, n)
  FOREACHVVVERTEX(bl2, w, m)
   if ((adist = w->squaredDistance(v))<mindist)
   {
	mindist=adist;
	*closest_on_bl1 = v;
	*closest_on_bl2 = w;
   }

 return mindist;
}

bool joinClosestComponents(ExtTriMesh *tin)
{
  Vertex *v,*w, *gv, *gw;
  Triangle *t, *s;
  Node *n;
  List triList, boundary_loops, *one_loop;
  List **bloops_array;
  int i, j, numloops;

  i=0;
  FOREACHVTTRIANGLE((&(tin->T)), t, n) t->info = NULL;
  FOREACHVTTRIANGLE((&(tin->T)), t, n) if (t->info == NULL)
  {
   i++;
   triList.appendHead(t);
   t->info = (void *)i;

   while(triList.numels())
   {
    t = (Triangle *)triList.popHead();
    if ((s = t->t1()) != NULL && s->info == NULL) {triList.appendHead(s); s->info = (void *)i;}
    if ((s = t->t2()) != NULL && s->info == NULL) {triList.appendHead(s); s->info = (void *)i;}
    if ((s = t->t3()) != NULL && s->info == NULL) {triList.appendHead(s); s->info = (void *)i;}
   }
  }

  if (i<2)
  {
   FOREACHVTTRIANGLE((&(tin->T)), t, n) t->info = NULL;
//   JMesh::info("Mesh is a single component. Nothing done.");
   return false;
  }

  FOREACHVTTRIANGLE((&(tin->T)), t, n)
  {
   t->v1()->info = t->v2()->info = t->v3()->info = t->info;
  }

  FOREACHVVVERTEX((&(tin->V)), v, n) if (!IS_VISITED2(v) && v->isOnBoundary())
  {
   w = v;
   one_loop = new List;
   do
   {
    one_loop->appendHead(w); MARK_VISIT2(w);
    w = w->nextOnBoundary();
   } while (w != v);
   boundary_loops.appendHead(one_loop);
  }
  FOREACHVVVERTEX((&(tin->V)), v, n) UNMARK_VISIT2(v);

  bloops_array = (List **)boundary_loops.toArray();
  numloops = boundary_loops.numels();

  int numtris = tin->T.numels();
  double adist, mindist=DBL_MAX;

  gv=NULL;
  for (i=0; i<numloops; i++)
   for (j=0; j<numloops; j++)
	if (((Vertex *)bloops_array[i]->head()->data)->info != ((Vertex *)bloops_array[j]->head()->data)->info)
	{
	 adist = closestPair(bloops_array[i], bloops_array[j], &v, &w);
	 if (adist<mindist) {mindist=adist; gv=v; gw=w;}
	}

  if (gv!=NULL) tin->joinBoundaryLoops(gv, gw, 1, 0, 0);

  FOREACHVTTRIANGLE((&(tin->T)), t, n) t->info = NULL;
  FOREACHVVVERTEX((&(tin->V)), v, n) v->info = NULL;

  free(bloops_array);
  while ((one_loop=(List *)boundary_loops.popHead())!=NULL) delete one_loop;

  return (gv!=NULL);
}

bool meshfix(
  const double epsilon_angle, 
  const bool keep_all_components, 
  ExtTriMesh & tin)
{
#ifndef MESHFIX_VERBOSE
  JMesh::quiet = true;
#endif
  if (epsilon_angle)
  {
    JMesh::acos_tolerance = asin((M_PI*epsilon_angle)/180.0);
#ifdef MESHFIX_VERBOSE
    printf("Fixing asin tolerance to %e\n",JMesh::acos_tolerance);
#endif
  }

  if (keep_all_components)
  {
#ifdef MESHFIX_VERBOSE
    printf("\nJoining input components ...\n");
#endif
    JMesh::begin_progress();
    while (joinClosestComponents(&tin)) JMesh::report_progress("Num. components: %d       ",tin.shells());
    JMesh::end_progress();
    tin.deselectTriangles();
  }

  // Keep only the biggest component
  int sc = tin.removeSmallestComponents();
  if (sc) JMesh::warning("Removed %d small components\n",sc);

  // Fill holes by taking into account both sampling density and normal field continuity
  tin.fillSmallBoundaries(tin.E.numels(), true, true);

  // Run geometry correction
  if (tin.boundaries() || !meshclean(epsilon_angle,tin))
  {
#ifdef MESHFIX_VERBOSE
    fprintf(stderr,"MeshFix failed!\n");
    fprintf(stderr,"Please try manually using ReMESH v1.2 or later (http://remesh.sourceforge.net).\n");
#endif
    return false;
  }
  return true;
}
