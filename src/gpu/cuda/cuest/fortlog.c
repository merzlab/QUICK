#include <stdio.h>

#include "quick_cuest.h"

void
cuest_debuglog (const char *str)
{
    fputs (str, quick_cuest_log_fp);
    fputc ('\n', quick_cuest_log_fp);
}
