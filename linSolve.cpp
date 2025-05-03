#include "Fluid.h"

void Fluid::Solver::GaussSedielRB(int maxIter, double dt, Fluid& f)
{
	const double pdxydt = f.density * f.dxy / dt * 1.9;
	for (int iter = 0; iter < maxIter; iter++)
	{
#pragma omp parallel for schedule(static)
		for (int i = 1; i < f.ny - 1; i++)
		{
#pragma omp simd
			for (int j = 1 + (i % 2); j < f.nx - 1; j += 2)
				GaussSedielKernel(i, j, pdxydt,f);
#pragma omp simd
            for (int j = 1 + (i + 1) % 2; j < f.nx - 1; j += 2)
                GaussSedielKernel(i, j, pdxydt,f);
		}
	}
}

void Fluid::Solver::GaussSedielKernel(int y, int x, double pdxydt, Fluid& f)
{
	if (f.barrier[y][x] == 0)return;
	double sx0 = f.barrier[y][x - 1], sx1 =f.barrier[y][x + 1];
	double sy0 = f.barrier[y - 1][x], sy1 = f.barrier[y + 1][x];
	double s = sx0 + sx1 + sy0 + sy1;
	if (s == 0)return;

	double div = f.u[y][x + 1] - f.u[y][x] + f.v[y + 1][x] - f.v[y][x];

	double p = -div / s;
    f.pressure[y][x] += pdxydt * p;

    f.u[y][x] -= sx0 * p, f.u[y][x + 1] += sx1 * p;
    f.v[y][x] -= sy0 * p, f.v[y + 1][x] += sy1 * p;
}

void Fluid::Solver::GaussSediel(int maxIter, double dt, Fluid& f)
{
	const double pdxydt = f.density * f.dxy / dt * 1.9;

	for (int iter = 0; iter < maxIter; iter++)
		for (int i = 1; i < f.ny - 1; i++)
			for (int j = 1; j < f.nx - 1; j++)
				GaussSedielKernel(i, j, pdxydt,f);
}

void Fluid::Solver::PCG_GSRB(int maxIter, double dt, Fluid& f)
{
    GaussSedielRB(maxIter * 0.2, dt, f);
    ConjugateGradient(maxIter * 0.8, dt, f);
}

void Fluid::Solver::ConjugateGradient(int maxIter, double dt, Fluid& f)
{
    const double pdxydt = f.density * f.dxy / dt;

    v2d b(f.ny, std::vector<double>(f.nx, 0.0));
    v2d sval(f.ny, std::vector<double>(f.nx, 0.0));

#pragma omp parallel for collapse(2)
    for (int i = 1; i < f.ny - 1; ++i) {
        for (int j = 1; j < f.nx - 1; ++j) {
            double sx0 = f.barrier[i][j - 1];
            double sx1 = f.barrier[i][j + 1];
            double sy0 = f.barrier[i - 1][j];
            double sy1 = f.barrier[i + 1][j];
            double s = sx0 + sx1 + sy0 + sy1;
            sval[i][j] = s;

            if (s == 0) {
                b[i][j] = 0.0;
                continue;
            }

            double div = f.u[i][j + 1] - f.u[i][j] + f.v[i + 1][j] - f.v[i][j];
            b[i][j] = -div;
        }
    }

    v2d p(f.ny, v1d(f.nx, 0.0)), r = b, d = b;

    double rr_old = DotProduct(r, r, f);
    if (rr_old < 1e-6) return;

    for (int iter = 0; iter < maxIter; ++iter) 
    {
        v2d Ad = MatrixVectorMultiply(d, sval, f);
        double alpha = rr_old / (DotProduct(d, Ad, f) + 1e-6);

#pragma omp parallel for collapse(2)
        for (int i = 1; i < f.ny - 1; ++i) 
        {
            for (int j = 1; j < f.nx - 1; ++j) 
            {
                p[i][j] += alpha * d[i][j];
                r[i][j] -= alpha * Ad[i][j];
            }
        }

        double rr_new = DotProduct(r, r, f);
        if (std::sqrt(rr_new) < 1e-6) break;

        double beta = rr_new / rr_old;

#pragma omp parallel for collapse(2)
        for (int i = 1; i < f.ny - 1; ++i) 
        {
            for (int j = 1; j < f.nx - 1; ++j) 
            {
                d[i][j] = r[i][j] + beta * d[i][j];
            }
        }

        rr_old = rr_new;
    }

    // 压力更新的并行化
#pragma omp parallel for collapse(2)
    for (int i = 1; i < f.ny - 1; ++i) {
        for (int j = 1; j < f.nx - 1; ++j) {
            f.pressure[i][j] += pdxydt * p[i][j];
        }
    }

    // 速度更新部分（注意存在数据竞争，保持串行）
    for (int i = 1; i < f.ny - 1; ++i) {
        for (int j = 1; j < f.nx - 1; ++j) {
            if (f.barrier[i][j] == 0) continue;

            double sx0 = f.barrier[i][j - 1];
            double sx1 = f.barrier[i][j + 1];
            double sy0 = f.barrier[i - 1][j];
            double sy1 = f.barrier[i + 1][j];
            double s = sx0 + sx1 + sy0 + sy1;
            if (s == 0) continue;

            double p_val = p[i][j];
            f.u[i][j] -= sx0 * p_val;
            f.u[i][j + 1] += sx1 * p_val;
            f.v[i][j] -= sy0 * p_val;
            f.v[i + 1][j] += sy1 * p_val;
        }
    }
}

v2d Fluid::Solver::MatrixVectorMultiply(const v2d& vec, const v2d& sval, Fluid& f)
{
    v2d result(f.ny, v1d(f.nx, 0.0));
#pragma omp parallel for collapse(2)
    for (int i = 1; i < f.ny - 1; ++i) {
        for (int j = 1; j < f.nx - 1; ++j) {
            double s = sval[i][j];
            double sum =
                vec[i][j - 1] * f.barrier[i][j - 1] +
                vec[i][j + 1] * f.barrier[i][j + 1] +
                vec[i - 1][j] * f.barrier[i - 1][j] +
                vec[i + 1][j] * f.barrier[i + 1][j];
            result[i][j] = s * vec[i][j] - sum * (s != 0);
        }
    }
    return result;
}

double Fluid::Solver::DotProduct(const v2d& a, const v2d& b, Fluid& f)
{
    double result = 0.0;
#pragma omp parallel for reduction(+:result) collapse(2)
    for (int i = 1; i < f.ny - 1; ++i) {
        for (int j = 1; j < f.nx - 1; ++j) {
            result += a[i][j] * b[i][j] * f.barrier[i][j];
        }
    }
    return result;
}