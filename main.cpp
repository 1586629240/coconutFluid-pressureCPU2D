#include "Fluid.h"
#include <omp.h>
#include <ctime>	
void setObstacle(double x, double y, bool reset,double t=0)
{
	double vx = (!reset) * ((x - scene.ballX) / scene.dt);
	double vy = (!reset) * ((y - scene.ballY) / scene.dt);

	scene.ballX = x, scene.ballY = y;
	
	auto f = scene.fluid;

	int nx = f->getNX() - 2, ny = f->getNY() - 2;

	for (int i = 1; i < ny; i++)
		for (int j = 1; j < nx; j++)
			f->getBrrier()[i][j] = 1.0;

	double r2 = scene.ballR * scene.ballR;
	for (int i = 1; i < ny; i++)
	{
		for (int j = 1; j < nx; j++)
		{
			double dx = (j + 0.5) * f->getdxy() - x;
			double dy = (i + 0.5) * f->getdxy() - y;

			if (dx * dx + dy * dy < r2)
			{
				f->getSmoke()[i][j] = (0.5 + 0.5 * sin(t));
				f->getBrrier()[i][j] = 0.0;
				f->getU()[i][j + 1] = f->getU()[i][j] = vx;
				f->getV()[i + 1][j] = f->getV()[i][j] = vy;
			}
		}
	}
}

void initScenePaint(size_t nxy)
{
	scene.dt = 0.01;
	scene.ballR = 0.05;

	double dxdy = 1.0 / nxy;
	int nx = nxy, ny = nxy;

	double density = 1000.0;
	auto f = scene.fluid = new Fluid(density, nx, ny, dxdy);

	scene.gravity = 0;
	scene.enableMouse = true;
}

void initSceneVortexStreet(size_t nxy)
{
	scene.dt = 0.01;
	scene.ballR = 0.05;

	double dxdy = 1.0 / nxy;
	int nx = nxy, ny = nxy;

	double density = 100000.0;
	auto f = scene.fluid = new Fluid(density, nx, ny, dxdy);

	double inVel = 2;
	for (int i = 0; i < f->getNY(); i++)
	{
		for (int j = 0; j < f->getNX(); j++)
		{
			double s = 1.0;
			if (i == 0 || j == 0 || i == ny + 1)s = 0.0;
			f->getBrrier()[i][j] = s;

			if (j == 1)f->getU()[i][j] = inVel;
		}
	}

	double pipeH = 0.05 * f->getNY();
	double minJ = floor(0.5 * f->getNY() - 0.5 * pipeH);
	double maxJ = floor(0.5 * f->getNY() + 0.5 * pipeH);

	for (int j = minJ; j < maxJ; j++)
		f->getSmoke()[j][0] = 1;

	setObstacle(0.2, 0.5, true);
	scene.gravity = 0.0;
	scene.enableMouse = true;
}

void simulate()
{
	scene.fluid->simulate(scene.dt, scene.gravity, 20);
}

void display(Fluid& fluid) {
    int N = fluid.getNY();
    auto buf = GetImageBuffer();
	
    for (int y = 0; y < 480; y++) {
        for (int x = 0; x < 640; x++) {
            int dx = static_cast<int>((x / 640.0f) * N);
            int dy = static_cast<int>((y / 480.0f) * N);
			float d = fluid.getSmoke()[dy][dx];
			buf[y * 640 + x] = HSVtoRGB(d*360, 1, 1);
			//buf[y * 640 + x] = RGB(d * 255, d * 255, d * 255);
        }
    }
    FlushBatchDraw();
}

void mouseEvent(Fluid& cube)
{
	if (scene.enableMouse == false) return;

	static bool drawing = false;
	static double t = 0;
	ExMessage m;
	if (peekmessage(&m)) 
	{
		switch (m.message) 
		{
		case WM_LBUTTONDOWN: drawing = true; break;
		case WM_LBUTTONUP:   drawing = false; break;
		}
	}

	if (drawing && m.message == WM_MOUSEMOVE) 
	{
		double x = m.x / 640.;
		double y = m.y / 480.;
		setObstacle(x, y, false, t);
		t += 0.001;
	}
}

int main()
{	
	initScenePaint(200);
	initgraph(640, 480);

	for (;;)
	{	
		simulate();
		mouseEvent(*scene.fluid);
		display(*scene.fluid);
	}

	delete scene.fluid;
	return 0;
}