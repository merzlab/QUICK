/*
 Copyright (C) 2006-2007 M.A.L. Marques

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include "util.h"

/*
  Lambert W function. 
  adapted from the Fortran code of Rickard Armiento

  Corless, Gonnet, Hare, Jeffrey, and Knuth (1996), 
         Adv. in Comp. Math. 5(4):329-359. 
*/

double LambertW(double z)
{
  double w;
  int i;

  /* Sanity check - function is only defined for z >= -1/e */
  if(z + 1.0/M_E < -10*DBL_EPSILON) {
    fprintf(stderr,"Error - Lambert function called with argument z = %e.\n",z);
    exit(1);
  } else if(z < -1.0/M_E)
    /* Value of W(x) at x=-1/e is -1 */
    return -1.0;
  
  /* If z is small, go with the first terms of the power expansion
     (if z smaller than cube root of epsilon, z^4 will be zero to
     machine precision).
   */
  if(fabs(z) < CBRT(DBL_EPSILON))
    return z - z*z + 1.5*z*z*z;

  /* Initial guess. */
  if(z <= -0.3140862435046707) { /* Point where sqrt and Taylor polynomials match */
    /* Near the branching point: first terms in eqn (4.22) */
    w = sqrt(2.0*M_E*z + 2.0) - 1.0;
    
  } else if(z <= 1.149876485041417) { /* Point where Taylor and log expansion match */

    /* Taylor series around origin */
    w = z - z*z + 1.5*z*z*z;

  } else {
    /* Asymptotic expansion */
    double lnz = log(z);

    w = lnz - log(lnz);
  }

  /* Find result through iteration */
  for(i=0; i<10; i++){
    double expmw, dw;
    expmw = exp(-w);
    
    /* Halley's equation, (5.9) in Corless et al */
    if( w != -1.0 )
      dw = - (w - z*expmw) / ( w + 1.0 - (w + 2.0)/(2.0*w + 2.0)*(w - z*expmw) );
    else
      dw = 0.0;

    w += dw;
    if(fabs(dw) < 10*DBL_EPSILON*(1.0 + fabs(w)))
      return w;
  }

  /* This should never happen! */
  fprintf(stderr, "%s\n%s\n", "lambert_w: iteration limit reached",
	  "Should never happen: execution aborted");
  exit(1);
}

/*
  Compute the dilogarithm, a form of spence-s function.

  based on the SLATEC routine by W. Fullerton
*/


static double pi26 = 1.644934066848226436472415166646025189219;
static double spencs[38] = 
  {
    +.1527365598892405872946684910028e+0,
    +.8169658058051014403501838185271e-1,
    +.5814157140778730872977350641182e-2,
    +.5371619814541527542247889005319e-3,
    +.5724704675185826233210603054782e-4,
    +.6674546121649336343607835438589e-5,
    +.8276467339715676981584391689011e-6,
    +.1073315673030678951270005873354e-6,
    +.1440077294303239402334590331513e-7,
    +.1984442029965906367898877139608e-8,
    +.2794005822163638720201994821615e-9,
    +.4003991310883311823072580445908e-10,
    +.5823462892044638471368135835757e-11,
    +.8576708692638689278097914771224e-12,
    +.1276862586280193045989483033433e-12,
    +.1918826209042517081162380416062e-13,
    +.2907319206977138177795799719673e-14,
    +.4437112685276780462557473641745e-15,
    +.6815727787414599527867359135607e-16,
    +.1053017386015574429547019416644e-16,
    +.1635389806752377100051821734570e-17,
    +.2551852874940463932310901642581e-18,
    +.3999020621999360112770470379519e-19,
    +.6291501645216811876514149171199e-20,
    +.9933827435675677643803887752533e-21,
    +.1573679570749964816721763805866e-21,
    +.2500595316849476129369270954666e-22,
    +.3984740918383811139210663253333e-23,
    +.6366473210082843892691326293333e-24,
    +.1019674287239678367077061973333e-24,
    +.1636881058913518841111074133333e-25,
    +.2633310439417650117345279999999e-26,
    +.4244811560123976817224362666666e-27,
    +.6855411983680052916824746666666e-28,
    +.1109122433438056434018986666666e-28,
    +.1797431304999891457365333333333e-29,
    +.2917505845976095173290666666666e-30,
    +.4742646808928671061333333333333e-31
  };


double xc_dilogarithm(const double x)
{
  const int nspenc = 38;
  double aux, dspenc;

  if (x > 2.0){
    aux = log(x);
    dspenc = 2.0*pi26 - 0.5*aux*aux;
    if(x < FLT_RADIX/DBL_EPSILON) 
      dspenc -= (1.0 + xc_cheb_eval(4.0/x - 1.0, spencs, nspenc))/x;

  }else if (x > 1.0){
    aux = x - 1.0;
    dspenc = pi26 - 0.5*log(x)*log(aux*aux/x)
      + aux*(1.0 + xc_cheb_eval(4.0*aux/x-1.0, spencs, nspenc))/x;

  }else if (x > 0.5){
     if (x != 1.0)
       dspenc = pi26 - log(x)*log(1.0 - x)
	 - (1.0 - x)*(1.0 + xc_cheb_eval(4.0*(1.0 - x)-1.0, spencs, nspenc));

  }else if (x >= 0.0){
    dspenc = x*(1.0 + xc_cheb_eval(4.0*x - 1.0, spencs, nspenc));

  }else if (x > -1.0){
    aux = log(1.0 - x);
    dspenc = -0.5*aux*aux - x*(1.0+ xc_cheb_eval(4.0*x/(x-1.0)-1.0, spencs, nspenc))/(x-1.0);

  }else{
    aux = log(1.0 - x);
    dspenc = -pi26 - 0.50*aux*(2.00*log(-x) - aux);

    if (x > -FLT_RADIX/DBL_EPSILON)
      dspenc += (1.0 + xc_cheb_eval(4.0/(1.0-x)-1.0, spencs, nspenc))/(1.0 - x);
  }

  return dspenc;
}
