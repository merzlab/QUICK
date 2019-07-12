#include "cuda_runtime.h"
#include "device_launch_parameters.h"

__device__ void gga_x_b88(param_struct * params, double rho, double sigma, double * zk, double * dfdr, double * dot){

  double gdm    = max(sqrt(sigma)/params->sfact, params->dens_threshold);
  double ds     = rho/params->sfact;
  double rhoLDA = pow(ds, params->alpha);
  double x    = gdm/pow(ds, params->beta2);

  double t1, t2, t3, t4, t5, t7, t8, t9;
  double t10, t11, t12, t15, t16;
  double f;


  t1 = params->M_CBRT3;
  t2 = t1 * t1;
  t3 = params->beta * t2;
  t4 = params->M_CBRT4;
  t5 = t3 * t4;
  t7 = cbrt(0.1e1 / 0.31415926535897932385e1);
  t8 = 0.1e1 / t7;
  t9 = x * x;
  t10 = t8 * t9;
  t11 = params->gamma * params->beta;
  t12 = log(x + sqrt(x * x + 0.1e1));
  t15 = t11 * x * t12 + 0.1e1;
  t16 = 0.1e1 / t15;
  f = 0.1e1 + 0.2e1 / 0.9e1 * t5 * t10 * t16;

  zk = rhoLDA * params->c_zk * f;
  zk /= rho;

  t20 = t8 * x;
  t24 = t15 * t15;
  t25 = 0.1e1 / t24;
  t27 = t9 + 0.1e1;
  t28 = sqrt(t27);
  t29 = 0.1e1 / t28;
  t32 = t11 * x * t29 + t11 * t12;
  t33 = t25 * t32;
  dfdx = 0.4e1 / 0.9e1 * t5 * t20 * t16 - 0.2e1 / 0.9e1 * t5 * t10 * t33;

        vrho[i] = (rhoLDA/ds)* (c_vrho0 * f + c_vrho1 * dfdx*x);

        if(gdm > params->dens_threshold){
                vsigma[i] = rhoLDA* (c_vsigma0 * dfdx*x/(2*sigma));
        }

}
