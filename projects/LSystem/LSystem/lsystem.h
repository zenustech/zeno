#include <iostream>
#include <stack>
#include "R3.h"
#include "turtle.h"
#include <string>
#include <map>
using namespace std;
typedef map<string,vector<string> > AssociativeArray;
class LSystem 
{
protected:
	R3Mesh * mesh;
	TurtleSystem turtle;
	void replaceAll(string& str, const string& from, const string& to) ;
    string produce(const string axiom, const AssociativeArray rules);
	virtual void run(const char command,const float param);
	float defaultCoefficient;
public:
	LSystem(R3Mesh *m)
	:mesh(m),turtle(mesh)
	{

	}
	string reproduce(const string axiom,const AssociativeArray rules, const int iterations=1);
	virtual string generateFromCode(const string code);
	void draw(const string data);
};