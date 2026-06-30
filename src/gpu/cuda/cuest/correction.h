#ifndef CUEST_CORRECTION_H
#define CUEST_CORRECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CORRECT_REORDER             1 // 0b001
#define CORRECT_NORM_CUEST_TO_QUICK 2 // 0b010
#define CORRECT_NORM_QUICK_TO_CUEST 6 // 0b110

// do not pass the following specifiers as a parameter
#define CORRECT_NORM_      2 // 0b010
#define CORRECT_FROMQUICK_ 4 // 0b100

void correct_o (double *o, uint8_t qspec);
void correct_C (double *C, size_t nocc, uint8_t qspec);

#endif
