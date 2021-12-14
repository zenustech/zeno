#include "lplus.h"
#include <fstream>
#include "R3Mesh.h"
using namespace std;
LPlusSystem::LPlusSystem(R3Mesh * m)
:LSystem(m),isPlus(false)
{
}

string LPlusSystem::generateFromCode(const string code, const bool plus)
{
	isPlus=plus;
	return LSystem::generateFromCode(code);

}
void LPlusSystem::run(const char command,const float param)
{
	if (!isPlus)
		return LSystem::run(command,param);

	float co=defaultCoefficient;
	float num=param;
	if (num==1)
		num*=co;
	switch (command)
	{
		case '+':
		turtle.turnLeft(num);
		break;
		case '-':
		turtle.turnRight(num);
		break;
		case '&':
		turtle.pitchDown(num);
		break;
		case '^':
		turtle.pitchUp(num);
		break;
		case '<':
		turtle.thicken(num);
		break;
		case '\\':
		turtle.rollLeft(num);
		break;
		case '/':
		////////////////////////////////////////  L++ ////////////////////////////////////////////////
		turtle.rollRight(num);
		break;
		case '>':
		turtle.narrow(num);
		break;
		case '%':
		turtle.setReduction(param);
		break;
		case '=':
		turtle.setThickness(param);
		break;
		case '|':
		turtle.turn180(param);
		break;
		case '*':
		turtle.drawLeaf(param);
		break;
		case 'F':
		case 'f':
		turtle.draw(param);
		case 'G':
		case 'g':
		turtle.move(param);
		break;
		case '[':
			turtle.save();
			break;
		case ']':
			turtle.restore();
			break;
		default:
		;
	}

}
