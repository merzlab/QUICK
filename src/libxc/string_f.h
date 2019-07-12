/*
 Copyright (C) 2003 M. Marques, A. Castro, A. Rubio, G. Bertsch

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

/* --------------------- Fortran to C string compatibility ---------------------- */
#include <stdlib.h>

#if defined(_CRAY)
#include <fortran.h>

#define to_c_str(f, c) {				    \
  char *fc; int slen;                                       \
  fc = _fcdtocp(f);                                         \
  for(slen=_fcdlen(f)-1; slen>=0 && fc[slen]==' '; slen--); \
  slen++;                                                   \
  c = (char *)malloc(slen+1);                               \
  strncpy(c, _fcdtocp(f), slen);                            \
  c[slen] = '\0';                                           \
}

#define to_f_str(c, f) {          \
  char *fc;  int flen, clen, i;   \
  flen = _fcdlen(f);              \
  fc = _fcdtocp(f);               \
  clen = strlen(c);               \
  for(i=0; i<clen && i<flen; i++) \
    fc[i] = c[i];                 \
  for(; i<flen; i++)              \
    fc[i] = ' ';                  \
}

#define STR_F_TYPE _fcd
#define TO_C_STR1(s, c) to_c_str(s, c)
#define TO_C_STR2(s, c) to_c_str(s, c)
#define TO_C_STR3(s, c) to_c_str(s, c)
#define TO_F_STR1(c, f) to_f_str(c, f)
#define TO_F_STR2(c, f) to_f_str(c, f)
#define TO_F_STR3(c, f) to_f_str(c, f)
#define STR_ARG1
#define STR_ARG2
#define STR_ARG3

#else

#define to_c_str(f, c, l) {		   \
  int i, ll;                               \
  ll = (int)l;                             \
  for(ll--; ll>=0; ll--)                   \
    if(f[ll] != ' ') break;                \
  ll++;                                    \
  c = (char *)malloc((ll+1)*sizeof(char)); \
  for(i=0; i<ll; i++) c[i] = f[i];         \
  c[i] = '\0';                             \
}

#define to_f_str(c, f, l) {                \
  int i, ll;                               \
  ll = (int)l;                             \
  for(i=0; i<ll && c[i]!='\0'; i++)        \
    f[i] = c[i];                           \
  for(; i<ll; i++)                         \
    f[i] = ' ';                            \
}


#define STR_F_TYPE char *
#define TO_C_STR1(s, c) to_c_str(s, c, l1)
#define TO_C_STR2(s, c) to_c_str(s, c, l2)
#define TO_C_STR3(s, c) to_c_str(s, c, l3)
#define TO_F_STR1(c, f) to_f_str(c, f, l1)
#define TO_F_STR2(c, f) to_f_str(c, f, l2)
#define TO_F_STR3(c, f) to_f_str(c, f, l3)
#define STR_ARG1     , unsigned long l1
#define STR_ARG2     , unsigned long l1, unsigned long l2
#define STR_ARG3     , unsigned long l1, unsigned long l2, unsigned long l3

#endif
