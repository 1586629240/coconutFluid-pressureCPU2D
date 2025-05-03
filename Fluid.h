#pragma once
#include "header.h"

class Fluid
{
	int nx, ny;
	double density, dxy;
	v2d pressure, smoke, newSmoke;
	v2d u, v, newU, newV, barrier;

	enum FieldTy { U, V, S };

public:

	v2d& getU();
	v2d& getV();
	v2d& getSmoke();
	v2d& getBrrier();
	v2d& getPressure();

	int getNX();
	int getNY();
	double getdxy();

	Fluid(double density, int NX, int NY, double h);

private:
	class Solver
	{
	public:
		static void PCG_GSRB(int maxIter, double dt, Fluid& f);
		static void GaussSediel(int maxIter, double dt, Fluid& f);
		static void GaussSedielRB(int maxIter, double dt, Fluid& f);
		static void ConjugateGradient(int maxIter, double dt, Fluid& f);

	private:
		static double DotProduct(const v2d& a, const v2d& b, Fluid& f);
		static void GaussSedielKernel(int y, int x, double pdxydt, Fluid& f);
		static v2d MatrixVectorMultiply(const v2d& vec, const v2d& sval, Fluid& f);
	};

	void boundryCondPeriod();
	void boundryCondNoSlip();
	void boundryCondDirichlet();

	double avgU(int i, int j);
	double avgV(int i, int j);

	void advectVel(double dt);
	void advectSmoke(double dt);
	void integrate(double dt, double gravity);
	double interpolation(double x, double  y, FieldTy field);

public:
	void simulate(double dt, double gravity, int numIters);
};

struct Scene
{
	Fluid* fluid = nullptr;
	double dt, gravity;
	double ballX, ballY, ballR;
	bool enableMouse = true;
};

extern Scene scene;