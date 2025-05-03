#include "Fluid.h"
#include <omp.h>
Scene scene;

v2d& Fluid::getU() { return u; }

v2d& Fluid::getV() { return v; }

v2d& Fluid::getSmoke() { return smoke; }

v2d& Fluid::getBrrier() { return barrier; }

v2d& Fluid::getPressure() { return pressure; }

int Fluid::getNX() { return nx; }

int Fluid::getNY() { return ny; }

double Fluid::getdxy() { return dxy; }

Fluid::Fluid(double density, int NX, int NY, double h)
{
	this->dxy = h;
	this->density = density;
	nx = NX + 2, ny = NY + 2;

	smoke = newSmoke = v2d(ny, v1d(nx, 0.0));
	u = v = newU = newV = barrier = pressure = v2d(ny, v1d(nx, 0.0));
}

void Fluid::integrate(double dt, double gravity)
{
	if (fabs(gravity) < 1e-6)return;
	for (int i = 1; i < ny - 1; i++)
		for (int j = 1; j < nx - 1; j++)v[i][j] += gravity * dt;
}

double Fluid::interpolation(double x, double y, FieldTy field)
{
	double h1 = 1.0 / dxy;
	double h2 = 0.5 * dxy;

	x = max(min(x, nx * dxy), dxy);
	y = max(min(y, ny * dxy), dxy);

	double dx = 0.0, dy = 0.0;

	v2d* f;

	switch (field) {
	case U: f = &u; dy = h2; break;
	case V: f = &v; dx = h2; break;
	case S: f = &smoke; dx = dy = h2; break;
	default: f = &u; break;
	}

	int x0 = min(floor((x - dx) * h1), nx - 1);
	double tx = ((x - dx) - x0 * dxy) * h1;
	int x1 = min(x0 + 1, nx - 1);

	int y0 = min(floor((y - dy) * h1), ny - 1);
	double ty = ((y - dy) - y0 * dxy) * h1;
	int y1 = min(y0 + 1, ny - 1);

	double sx = 1.0 - tx;
	double sy = 1.0 - ty;

	double val = sx * sy * (*f)[y0][x0] + sx * ty * (*f)[y1][x0] + tx * ty * (*f)[y1][x1] + tx * sy * (*f)[y0][x1];
	return val;
}

double Fluid::avgU(int y, int x) {
	double retu = (u[y - 1][x] + u[y][x] + u[y - 1][x + 1] + u[y][x + 1]) * 0.25;
	return retu;

}

double Fluid::avgV(int y, int x) {
	double retv = (v[y][x - 1] + v[y][x] + v[y + 1][x - 1] + v[y + 1][x]) * 0.25;
	return retv;
}

void Fluid::advectVel(double dt) {

	newU = u; newV = v;
	double dxy2 = 0.5 * dxy;

	for (int i = 1; i < ny; i++)
	{
		for (int j = 1; j < nx; j++)
		{
			if (barrier[i][j] != 0.0 && barrier[i][j - 1] != 0.0 && i < ny - 1) {
				double y = i * dxy + dxy2, x = j * dxy;
				double valu = u[i][j], v = avgV(i, j);

				x = x - dt * valu, y = y - dt * v;
				valu = interpolation(x, y, U); newU[i][j] = valu;
			}
			if (barrier[i][j] != 0.0 && barrier[i - 1][j] != 0.0 && j < nx - 1) {
				double y = i * dxy, x = j * dxy + dxy2;
				double u = avgU(i, j), valv = v[i][j];

				x = x - dt * u, y = y - dt * valv;
				valv = interpolation(x, y, V); newV[i][j] = valv;
			}
		}
	}

	u = newU; v = newV;
}

void Fluid::advectSmoke(double dt)
{
	newSmoke = smoke;
	double dxy2 = 0.5 * dxy;

	for (int i = 1; i < ny - 1; i++) {
		for (int j = 1; j < nx - 1; j++) {

			if (barrier[i][j] != 0.0) {
				double valv = (v[i][j] + v[i + 1][j]) * 0.5;
				double valu = (u[i][j] + u[i][j + 1]) * 0.5;
				double y = i * dxy + dxy2 - dt * valv;
				double x = j * dxy + dxy2 - dt * valu;

				newSmoke[i][j] = interpolation(x, y, S);
			}
		}
	}
	smoke = newSmoke;
}

void Fluid::simulate(double dt, double gravity, int numIters)
{
	integrate(dt, gravity);

	for (int i = 0; i < ny; i++)
		for (int j = 0; j < nx; j++)pressure[i][j] = 0;

	Solver::PCG_GSRB(numIters, dt, *this);

	boundryCondNoSlip();
	advectVel(dt);
	advectSmoke(dt);
}
