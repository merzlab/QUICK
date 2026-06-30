#ifndef CUEST_CORRECTION_H
#define CUEST_CORRECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CORRECT_REORDER          1 // 0b01
#define CORRECT_NORM             2 // 0b10
#define CORRECT_REORDER_AND_NORM 3 // 0b11

void correct_o (double *o, uint8_t qspec);
void correct_C (double *C, size_t nocc, uint8_t qspec);

#endif
