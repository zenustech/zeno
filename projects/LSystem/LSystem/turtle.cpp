#ifdef _WIN32
#define _USE_MATH_DEFINES
#endif

#include <iostream>
#include "turtle.h"
Turtle::Turtle()
:position(0,0,0)
,direction(0,1,0) //default direction is towards Y, stomach is faced -Z
,right(1,0,0)
,thickness(1)
,reduction(.95)

{

}
void Turtle::setReduction(float param)
{
  reduction=param/100;
}
void Turtle::setThickness(float param)
{
  thickness=param/100;
}
void Turtle::thicken(float param)
{
  thickness+=thickness * param/100;
}
void Turtle::narrow(float param)
{
  thicken(-param);
}
void Turtle::turnRight(float angle)
{
  // cout <<"Turning "<<angle<<" right"<<endl;
  angle=angle*M_PI/180;
  R3Vector axis=direction;
  axis.Cross(right);
  direction.Rotate(axis,angle);
  right.Rotate(axis,angle);
  direction.Normalize();
  right.Normalize();
}
void Turtle::turnLeft(float angle)
{
  turnRight(-angle);
}
void Turtle::pitchUp(float angle)
{
  // cout <<"Pitching "<<angle<<" up"<<endl;
  angle=angle*M_PI/180;
  direction.Rotate(right,angle);
  direction.Normalize();
}
void Turtle::pitchDown(float angle)
{
  pitchUp(-angle);
}
void Turtle::rollRight(float angle)
{
  // cout <<"Rolling "<<angle<<" right"<<endl;
  angle=angle*M_PI/180;
  right.Rotate(direction,angle);
  right.Normalize();
}
void Turtle::rollLeft(float angle)
{
  rollRight(-angle);
}
void Turtle::move(float distance)
{
  R3Vector t=direction;
  t.Normalize();
  position+=distance*t;
}
void Turtle::turn180(float temp)
{
  turnRight(M_PI);
}
TurtleSystem::TurtleSystem(R3Mesh * m)
:mesh(m)
{
}
void TurtleSystem::save()
{
  Turtle t=(Turtle)*this;
  // for (int i=0;i<state.size();++i) cout <<"--";
  // cout <<"Saving state ("<<t.position.X()<<","<<t.position.Y()<<","<<t.position.Z()<<")-("
  //     <<t.direction.X()<<","<<t.direction.Y()<<","<<t.direction.Z()<<")"<<endl;
  state.push(t);
}
void TurtleSystem::restore()
{
  Turtle t=state.top();

  state.pop();
  // for (int i=0;i<state.size();++i) cout <<"--";
  // cout <<"Restoring state ("<<t.position.X()<<","<<t.position.Y()<<","<<t.position.Z()<<")-("
      // <<t.direction.X()<<","<<t.direction.Y()<<","<<t.direction.Z()<<")"<<endl;
  position=t.position;
  direction=t.direction;
  right=t.right;
  thickness=t.thickness;
  reduction=t.reduction;
  // (Turtle)*this=t; //FIXME: doesn't work, figure out why!
}
void TurtleSystem::drawLeaf(float param)
{

  R3Shape s=mesh->Leaf(direction);

  mesh->ScaleShape(s,param,param,param);
  R3Vector leafDirection(0,1,0);
  R3Vector axis=leafDirection %  direction; 
  axis.Normalize();
  if (!(fabs(axis.X())<.001 && fabs(axis.Y())<.001 && fabs(axis.Z())<.001) )
  {
    double rotateAngle=acos(leafDirection.Dot(direction)/( leafDirection.Length() * direction.Length() ) );
    if (fabs(rotateAngle)>.001)
      mesh->RotateShape(s,rotateAngle,axis);
  }
  mesh->TranslateShape(s,position.X(),position.Y(),position.Z());
}
void TurtleSystem::draw(float param)
{
  static int num=0;
  if (num++ % 1000 ==0) cout <<num<<" drawing"<<endl;

  int slices;
  if (thickness<.2)
    slices=20;
  else if (thickness<.4)
    slices=40;
  else if (thickness<.6)
    slices=60;
  else if (thickness<.8)
    slices=80;
  else
    slices=100;
  R3Shape s=mesh->Cylinder(reduction,slices);

  mesh->ScaleShape(s,param*thickness,param,param*thickness);
  R3Vector cylinderDirection(0,1,0);
  R3Vector axis=cylinderDirection %  direction; //the axis to rotate on
  axis.Normalize();
  // printf("Direction: %f %f %f\n",direction.X(),direction.Y(),direction.Z());
  // printf("Shape Dir: %f %f %f\n",cylinderDirection.X(),cylinderDirection.Y(),cylinderDirection.Z());
  // printf("Rotation Axis: %f %f %f\n",axis.X(),axis.Y(),axis.Z());
  if (!(fabs(axis.X())<.001 && fabs(axis.Y())<.001 && fabs(axis.Z())<.001) )
  {

    double rotateAngle=acos(cylinderDirection.Dot(direction)/( cylinderDirection.Length() * direction.Length() ) );
    if (fabs(rotateAngle)>.001)
    {
      // printf("Rotating by %f.\n",rotateAngle*180);
      mesh->RotateShape(s,rotateAngle,axis);
    }
    
  }

  mesh->TranslateShape(s,position.X(),position.Y(),position.Z());

}
