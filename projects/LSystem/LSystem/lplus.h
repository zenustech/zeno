#include <iostream>
#include <stack>
#include "R3.h"
#include "turtle.h"
#include <string>
#include <map>
#include "lsystem.h"
using namespace std;
class LPlusSystem: public LSystem
{
public:
	bool isPlus;
	LPlusSystem(R3Mesh *m);
	string generateFromCode(const string code, const bool plus);
	virtual void run(const char command,const float param);
};