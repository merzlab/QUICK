#ifndef CUEST_CORRECTION_H
#define CUEST_CORRECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CORRECT_REORDER 1 // 0b01
#define CORRECT_NORM    2 // 0b10

#define CORRECT_CUEST_TO_QUICK 0
#define CORRECT_QUICK_TO_CUEST 1

/**
 * @param[in] `qspec` Takes the following specifiers that can be combined with bitwise OR (`|`):
 *   - CORRECT_REORDER; reorders `o` between QUICK and cuEST basis set ordering
 *   - CORRECT_NORM_CUEST_TO_QUICK; corrects normalization when `o` is given in cuEST convention
 *   - CORRECT_NORM_QUICK_TO_CUEST; corrects normalization when `o` is given in QUICK convention
 */
void correct_o (double *o, int8_t qspec, int8_t dirspec);

/**
 * @param[in] `nocc` Number of occupied orbitals
 *
 * @param[in] `qspec` Takes the following specifiers that can be combined with bitwise OR (`|`):
 *   - CORRECT_REORDER; reorders `o` between QUICK and cuEST basis set ordering
 *   - CORRECT_NORM_CUEST_TO_QUICK; corrects normalization when `o` is given in cuEST convention
 *   - CORRECT_NORM_QUICK_TO_CUEST; corrects normalization when `o` is given in QUICK convention
 */
void correct_C (double *C, size_t nocc, int8_t qspec, int8_t dirspec);

#endif
