#ifndef CUEST_CORRECTION_H
#define CUEST_CORRECTION_H

#include <stdbool.h>
#include <stddef.h>

void correct_o (double *o, bool norm);
void reorder_PC (double *C, size_t nocc);

#endif
