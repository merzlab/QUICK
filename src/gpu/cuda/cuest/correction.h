#ifndef CUEST_CORRECTION_H
#define CUEST_CORRECTION_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define CORRECT_REORDER             0x1 // 0b0001
#define CORRECT_NORM_CUEST_TO_QUICK 0x2 // 0b0010
#define CORRECT_NORM_QUICK_TO_CUEST 0x6 // 0b0110
#define CORRECT_NORM_INV            0x8 // 0b1000

// do not pass the following specifiers as a parameter
#define CORRECT_NORM_      0x2 // 0b0010
#define CORRECT_FROMQUICK_ 0x4 // 0b0100

/** Must be called after initializing main basis set */
void cuest_init_correct ();
void cuest_deinit_correct ();

/**
 * @param[in] `qspec` Takes the following specifiers that can be combined with bitwise OR (`|`):
 *   - CORRECT_REORDER; reorders `o` between QUICK and cuEST basis set ordering
 *   - CORRECT_NORM_CUEST_TO_QUICK; corrects normalization when `o` is given in cuEST convention
 *   - CORRECT_NORM_QUICK_TO_CUEST; corrects normalization when `o` is given in QUICK convention
 *   - CORRECT_NORM_INV; inverts normalization coefficients
 */
void cuest_correct_o (double *o, int8_t qspec);

void cuest_correct_P (double *o, int8_t qspec);

/**
 * @param[in] `nocc` Number of occupied orbitals
 *
 * @param[in] `qspec` Takes the following specifiers that can be combined with bitwise OR (`|`):
 *   - CORRECT_REORDER; reorders `o` between QUICK and cuEST basis set ordering
 *   - CORRECT_NORM_CUEST_TO_QUICK; corrects normalization when `o` is given in cuEST convention
 *   - CORRECT_NORM_QUICK_TO_CUEST; corrects normalization when `o` is given in QUICK convention
 *   - CORRECT_NORM_INV; inverts normalization coefficients
 */
void cuest_correct_C (double *C, size_t nocc, int8_t qspec);

#endif
