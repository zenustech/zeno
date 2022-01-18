#ifndef TURTLE_H
#define TURTLE_H
#include <stack>
#include "R3.h"
#include "R3Mesh.h"
using namespace std;
struct Turtle 
{
  R3Vector position;
  R3Vector direction;
  R3Vector right;
  float thickness;
  float reduction;
  Turtle();
  void turnRight(float angle);
  void turnLeft(float angle);
  void pitchDown(float angle);
  void pitchUp(float angle);
  void rollLeft(float angle);
  void rollRight(float angle);
  void move(float distance);
  void turn180(float temp);
  void thicken(float param);
  void narrow(float param);
  void setThickness(float param);
  void setReduction(float param);


};
class TurtleSystem: public Turtle 
{
  stack<Turtle> state;
  R3Mesh *mesh;
public:
  TurtleSystem(R3Mesh * m);
  void save();
  void restore();
  void draw(float param);
  void drawLeaf(float param);

};

#endif