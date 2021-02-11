/*
 * Copyright (c) 2014 NVIDIA Corporation. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 *
 *   Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 *
 *   Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 *
 *   Neither the name of NVIDIA Corporation nor the names of its contributors
 *   may be used to endorse or promote products derived from this software 
 *   without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#if !defined(BOYS_FUNCTIONS_H_)
#define BOYS_FUNCTIONS_H_

#if defined(__cplusplus)
extern "C" {
#endif /* __cplusplus */

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#error This code requires compute capability 2.0 or higher
#endif

#include <math.h>  /* import exp(), pow(), erf(), ... */

/* Gamma (n+0.5), n=0...50 */
__constant__ double gamtab[51] =
{
    1.7724538509055161e+000,
    8.8622692545275805e-001,
    1.3293403881791370e+000,
    3.3233509704478426e+000,
    1.1631728396567448e+001,
    5.2342777784553519e+001,
    2.8788527781504433e+002,
    1.8712543057977884e+003,
    1.4034407293483413e+004,
    1.1929246199460901e+005,
    1.1332783889487856e+006,
    1.1899423083962249e+007,
    1.3684336546556586e+008,
    1.7105420683195732e+009,
    2.3092317922314240e+010,
    3.3483860987355646e+011,
    5.1899984530401250e+012,
    8.5634974475162062e+013,
    1.4986120533153360e+015,
    2.7724322986333720e+016,
    5.4062429823350752e+017,
    1.1082798113786905e+019,
    2.3828015944641842e+020,
    5.3613035875444143e+021,
    1.2599063430729375e+023,
    3.0867705405286966e+024,
    7.8712648783481775e+025,
    2.0858851927622668e+027,
    5.7361842800962338e+028,
    1.6348125198274267e+030,
    4.8226969334909086e+031,
    1.4709225647147272e+033,
    4.6334060788513905e+034,
    1.5058569756267020e+036,
    5.0446208683494509e+037,
    1.7403941995805607e+039,
    6.1783994085109903e+040,
    2.2551157841065116e+042,
    8.4566841903994186e+043,
    3.2558234133037758e+045,
    1.2860502482549915e+047,
    5.2085035054327160e+048,
    2.1615289547545769e+050,
    9.1864980577069524e+051,
    3.9961266551025243e+053,
    1.7782763615206234e+055,
    8.0911574449188361e+056,
    3.7623882118872588e+058,
    1.7871344006464480e+060,
    8.6676018431352724e+061,
    4.2904629123519598e+063
};

static __forceinline__ __device__ double my_fast_scale (double a, int i)
{
  unsigned int ihi, ilo;
  ilo = __double2loint (a);
  ihi = __double2hiint (a);
  return __hiloint2double ((i << 20) + ihi, ilo);
}

static __forceinline__ __device__ double my_fast_sqrt(double a)
{
  double x;
  asm ("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(x) : "d"(a));
  /* A. Schoenhage's coupled iteration for the square root, as described in:
     Timm Ahrendt, Schnelle Berechnungen der Exponentialfunktion auf hohe
     Genauigkeit. Berlin, Logos 1999
  */
  double v = my_fast_scale (x, -1);
  double w = a * x;
  w = __fma_rn (__fma_rn (w, -w, a), v, w);
  v = __fma_rn (__fma_rn (x, -w, 1.0), v, v);
  w = __fma_rn (__fma_rn (w, -w, a), v, w);
  return w;
}

static __forceinline__ __device__ double my_fast_rsqrt (double a)
{
  double r, e, t;
  asm ("rsqrt.approx.ftz.f64 %0, %1;" : "=d"(r) : "d"(a));
  t = __dmul_rn (r, r);
  e = __fma_rn (a, -t, 1.0);
  t = __fma_rn (0.375, e, 0.5);
  e = __dmul_rn (e, r);
  r = __fma_rn (t, e, r);
  return r;
}

static __forceinline__ __device__ double my_fast_rcp (double a) 
{ 
  double e, y; 
  asm ("rcp.approx.ftz.f64 %0,%1;" : "=d"(y) : "d"(a));
  e = __fma_rn (-a, y, 1.0); 
  e = __fma_rn ( e, e, e); 
  y = __fma_rn ( e, y, y); 
  return y; 
} 

static __forceinline__ __device__ double boys0_core (double a)
{
  double r;
  r = erf (my_fast_sqrt (a)) * my_fast_rsqrt (a);
  r = fma (r, 8.8622692545275805e-01, r * -3.8332932499128993e-17);
  if (isnan (a)) r = a + a;                           // NaNs
  if (a < 0.0) r = __hiloint2double (0xfff80000, 0);  // negative arguments
  if (a == __hiloint2double (0x7ff00000, 0)) r = 0.0; // infinities
  return r;
}

/* Boys function of order 0. It is closely related to the error function [1]: 
   F_0(x) = sqrt (pi / 4*x) * erf (sqrt(x))

   Special case handling: For arguments < 0.0, the NaN encoding INDEFINITE is
   returned.

   [1] S. Reine, T. Helgaker, and R. Lindh, Multi-electron integrals. Wiley 
   Interdisciplinary Reviews: Computational Molecular Science, Vol. 2, No. 2,
   March/April 2012, pp. 290-303
*/
static __device__ double boys0 (double a)
{
  double r;
  r = boys0_core (a);
  if ((a >= 0.0) && (__double2hiint (a) < 0x3ca80000)) r = 1.0; // small args
  return r;
}

/* Boys function of order 1. This uses a minimax polynomial approximation for
   small arguments, and one step of the forward recurrence [1]

   F(m,x) = (1 / 2*x) * ((2*m-1) * F(m-1,x) - exp(-x))

   for large arguments, using the Boys function of order zero as the base. The
   well-known numerical issues with the forward recurrence are mitigated by the
   use of fused multiply-add (FMA).

   Special case handling: For arguments < 0.0, the NaN encoding INDEFINITE is
   returned.

   [1] I.I. Guseinov and B.A. Mamedov, Evaluation of the Boys function using
       analytical relatios. Journal of Mathematical Chemistry, Vol. 40, No. 2,
       August 2006, pp. 179-183
*/
static __device__ double boys1 (double a)
{
  double r;
  if ((a >= 0.0) && (__double2hiint (a) < 0x40010000)) { // < 2.125
    r =             1.1970946332641065E-013;
    r = fma (r, a, -3.8586551297453333E-012);
    r = fma (r, a,  6.9894184117879818E-011);
    r = fma (r, a, -9.7933814582289743E-010);
    r = fma (r, a,  1.1931735072024966E-008);
    r = fma (r, a, -1.3114658399573355E-007);
    r = fma (r, a,  1.3052558861168618E-006);
    r = fma (r, a, -1.1671259420622283E-005);
    r = fma (r, a,  9.2592547606971200E-005);
    r = fma (r, a, -6.4102562259366405E-004);
    r = fma (r, a,  3.7878787829159679E-003);
    r = fma (r, a, -1.8518518517707788E-002);
    r = fma (r, a,  7.1428571428501481E-002);
    r = fma (r, a, -1.9999999999999760E-001);
    r = fma (r, a,  3.3333333333333331E-001);
  } else {
    r = 1.0 / (a + a);
    r = fma (boys0_core (a), r, -r * exp (-a));
  }
  return r;
}

/* Boys function of order 2. This uses a minimax polynomial approximation for
   small arguments, and two steps of the forward recurrence [1]

   F(m,x) = (1 / 2*x) * ((2*m-1) * F(m-1,x) - exp(-x))

   for large arguments, using the Boys function of order zero as the base. The
   well-known numerical issues with the forward recurrence are mitigated by the
   use of fused multiply-add (FMA).

   Special case handling: For arguments < 0.0, the NaN encoding INDEFINITE is
   returned.

   [1] I.I. Guseinov and B.A. Mamedov, Evaluation of the Boys function using
       analytical relatios. Journal of Mathematical Chemistry, Vol. 40, No. 2,
       August 2006, pp. 179-183
*/
static __device__ double boys2 (double a)
{
  double r;
  if ((a >= 0.0) && (__double2hiint (a) < 0x40010000)) { // < 2.125
    r =             1.1104955945072385E-013;
    r = fma (r, a, -3.5854166844010486E-012);
    r = fma (r, a,  6.4885794420476957E-011);
    r = fma (r, a, -9.0593405070342565E-010);
    r = fma (r, a,  1.0974611729014906E-008);
    r = fma (r, a, -1.1973717787771906E-007);
    r = fma (r, a,  1.1809379722172225E-006);
    r = fma (r, a, -1.0442697679432980E-005);
    r = fma (r, a,  8.1699300797023348E-005);
    r = fma (r, a, -5.5555553662436290E-004);
    r = fma (r, a,  3.2051281999578218E-003);
    r = fma (r, a, -1.5151515150657109E-002);
    r = fma (r, a,  5.5555555555480210E-002);
    r = fma (r, a, -1.4285714285714021E-001);
    r = fma (r, a,  1.9999999999999998E-001);
  } else {
    double ba = boys0_core (a);
    double ta = a + a;
    double ra = 1.0 / ta;
    double ea = exp (-a);
    r = fma (ba, ra, -ra * ea);
    r = fma (r + r, ra, fma (ra, -ea, r * ra));
  }
  return r;
}

/* Boys function of order 3. This uses a minimax polynomial approximation for
   small arguments, and three steps of the forward recurrence [1]

   F(m,x) = (1 / 2*x) * ((2*m-1) * F(m-1,x) - exp(-x))

   for large arguments, using the Boys function of order zero as the base. The
   well-known numerical issues with the forward recurrence are mitigated by the
   use of fused multiply-add (FMA).

   Special case handling: For arguments < 0.0, the NaN encoding INDEFINITE is
   returned.

   [1] I.I. Guseinov and B.A. Mamedov, Evaluation of the Boys function using
       analytical relatios. Journal of Mathematical Chemistry, Vol. 40, No. 2,
       August 2006, pp. 179-183
*/
static __device__ double boys3 (double a)
{
  double r;
  if ((a >= 0.0) && (__double2hiint (a) < 0x4003c200)) { // < 2.4697265625
    r =            -5.2902638238348148E-015;
    r = fma (r, a,  2.0097738177575676E-013);
    r = fma (r, a, -4.1563023331237324E-012);
    r = fma (r, a,  6.4533728874869425E-011);
    r = fma (r, a, -8.5578480573758601E-010);
    r = fma (r, a,  1.0189261416781104E-008);
    r = fma (r, a, -1.1020221508200681E-007);
    r = fma (r, a,  1.0782982060549489e-006);
    r = fma (r, a, -9.4481965040540530E-006);
    r = fma (r, a,  7.3099398352704758E-005);
    r = fma (r, a, -4.9019607116187254E-004);
    r = fma (r, a,  2.7777777756999018E-003);
    r = fma (r, a, -1.2820512820149871E-002);
    r = fma (r, a,  4.5454545454511754E-002);
    r = fma (r, a, -1.1111111111110986E-001);
    r = fma (r, a,  1.4285714285714285E-001);
  } else {
    double ba = boys0_core (a);
    double ta = a + a;
    double ra = 1.0 / ta;
    double ea = exp (-a);
    r = fma (ba, ra, -ra * ea);
    r = fma (r + r, ra, fma (ra, -ea, r * ra));
    r = fma (4.0 * r, ra, fma (ra, -ea, r * ra));
  }
  return r;
}

/* The Boys function F_m(a) is a definite integral encountered in molecular
   calculations with a Gaussian basis [1]. It is closely related to the lower
   incomplete Gamma function [2]. The following implementation can compute the
   Boys functions of orders 0 through 50, for real argument in the positive 
   half-plane. Results are accurate to almost 15 decimal digits, please refer
   to the file boys_functions_accuracy.txt for more detailed information.

   Special case handling: For arguments < 0.0, or orders < 0 or > 50, the NaN
   encoding INDEFINITE is returned.

   [1] S.F. Boys, Electronic Wave Functions. I. A General Method of Calculation
       for the Stationary States of Any Molecular System, Proceedings Royal 
       Society London A, vol. 200, 1950, pp. 542-554.
   [2] I.I. Guseinov and B.A. Mamedov, Evaluation of the Boys function using
       analytical relatios. Journal of Mathematical Chemistry, Vol. 40, No. 2,
       August 2006, pp. 179-183
*/
static __device__ double boys (int m, double a)
{
  double r;
  if ((m < 0) || (m > 50)) {
    r = __hiloint2double (0xfff80000, 0x00000000); // NaN INDEFINITE
  } else if (m < 2) {
    if (m == 0) {
      r = boys0 (a);
    } else {
      r = boys1 (a);
    }
  } else if (m < 4) {
    if (m == 2) {
      r = boys2 (a);
    } else {
      r = boys3 (a);
    }
  } else {
    double mph = m + 0.5;
    double ea = exp (-a);
    r = rsqrt (a);
    if (a > (mph + 1.0)) {
      double g, p, oop, lig;

      g = gamtab[m]; // Gamma (m+0.5)
      p = pow (a, -0.5 * mph);
      oop = my_fast_rcp (p);

      if ((float)a < ((7.6675415e-3f * m + 2.2382813f) * m + 42.841797f)) {
        /*
          Use Robert Israel's formula for the upper incomplete gamma function 
          of half integer orders

          http://math.stackexchange.com/questions/724068/expressing-upper-incomplete-gamma-function-of-half-integer-order-in-terms-of-gam
      
          upper_incomplete_gamma (m+0.5,a) = Gamma (m+0.5) * erfc(sqrt(a)) +

                                 m-1
          a**(m-0.5) * exp(-a) + sum (pochhammer (0.5-m,k) / -a**k)
                                 k=0

          lower_incomplete_gamma (m+0.5,a) = Gamma (m+0.5) - 
                                             upper_incomplete_gamma (m+0.5,a)

          boys(m,a) = 1/(2*a**(m+0.5)) * lower_incomplete_gamma (m+0.5,a)
        */
        double sum = 1.0;
        double term = 1.0;
        double ooa = my_fast_rcp (a);
        double mphooa = mph * ooa;
        double u, e;
        
        for (int k = 1; k < m; k++) {
          term = term * fma (k, -ooa, mphooa);
          sum += term;
        }

        u = ooa * ea * sum * oop * oop;
        e = erf (my_fast_sqrt (a));
        lig = fma (g, e, -u); // lower_incomplete_gamma (m+0.5,a)
      } else {
        lig = g; // lower_incomplete_gamma (m+0.5,a)
      }
      r = lig * 0.5 * p * p;
    } else if (a >= 0) {
      /* Use series expansion for small arguments, based on series expansion
         for the lower incomplete gamma function:

         A. Erdelyi, W. Magnus, F. Oberhettinger, and F.G. Tricomi, Higher 
         Transcendental Functions, Vol. 2. New York, NY: McGraw-Hill 1953

                   inf
      gamma(z,a) = sum (a**z * a**n * exp(-a) / (z * (z+1) * .. * (z+n)))
                   n=0
                                    inf
      gamma(z,a) = a**z * exp(-a) * sum (a**n / (z * (z+1) * .. * (z+n)))
                                    n=0
                                               inf
      gamma(z,a) = a**z * 1/z * exp(-a) * [1 + sum (a**n / ((z+1)* .. *(z+n)))]
                                               n=1

      boys(m,a) = 1/(2 * a**(m+0.5)) * gamma (m+0.5, a);   set z = m+0.5, then

                                            inf
      boys(m,a) = 1/2 * 1/z *exp(-a) * [1 + sum (a**n / ((z+1) * .. * (z+n)))]
                                            n=1
      therefore:

      z = m+0.5
 
      r = 1/2 * 1/z * exp(-a)
    
          inf
      s = sum (a**n / ((z+1) * ... * (z+n)))
          n=1

      boys(m,a) = fma (s, r, r)

      */
      double z = mph;
      double s = 0.0;
      double t;
      r = 0.5 * my_fast_rcp (z) * ea;
      z += 1.0;
      t = a * my_fast_rcp (z);

      while (__double2hiint(t) >= 0x3c600000) { // >= 2**-57
        z += 1.0;
        s += t;
        t *= a * my_fast_rcp (z);
      }

      r = fma (s, r, r);
    }    
  }
  return r;
}

#if defined(__cplusplus)
}
#endif /* __cplusplus */

#endif /* BOYS_FUNCTIONS_H_ */
