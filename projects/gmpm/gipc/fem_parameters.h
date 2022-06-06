#pragma once
#ifndef FEM_PARAMETERS_H
#define FEM_PARAMETERS_H

namespace FEM {
	const static double PI = 3.14159265358979323846264338327950288419716939937510582097494459230781640628620899;
	const static double density = 1e3;
	const static double YoungModulus = 1e4;
	const static double PoissonRate = 0.49;
	//const static double explicit_time_step = 0.000001;
	//const static double implicit_time_step = 0.001;
	const static double lengthRateLame = YoungModulus / (2 * (1 + PoissonRate));
	const static double volumeRateLame = YoungModulus * PoissonRate / ((1 + PoissonRate) * (1 - 2 * PoissonRate));
	const static double lengthRate = 4 * lengthRateLame / 3;
	const static double volumeRate = volumeRateLame + 5 * lengthRateLame / 6;
	const static double frictionRate = 0.2;
	//const static double aniosScale = lengthRate * 100;
	//const static double iosToAnios = lengthRateLame * 1;
	//static float contract_ratio = 1;
	//static int expression_Count = 0;
	//const static bool faceAnimation = false;
	//const static float iosStiff = 1;
	//const static float aniosStiff = 1;
	//const static bool isRehabitate = false;
	//const static double rehabit_threshold = 1e-7;
	//static double Hhat = 1e-6;
	//const static double IPC_dt = 0.025;
	//static double Kappa = 0;
	//const static double IPC_Stiffness = 1e3;
	//const static bool addGrivity = true;
	//const static int numSp = 154;//fldm.points.size();
	//const static int numMsc = 19;
	//static int step_counts = 0;
	//static int exit_counts = 0;
}


#endif // !FEM_PARAMETERS_H

