#include <stdio.h>
/*
 * EmbedIA - Embedded Machine Learning and Neural Networks Framework
 * Copyright (c) 2022
 * César Estrebou & contributors
 * Instituto de Investigación en Informática LIDI (III-LIDI)
 * Facultad de Informática - Universidad Nacional de La Plata (UNLP)
 * Originally developed with student contributions
 *
 * Licensed under the BSD 3-Clause License. See LICENSE file for details.
 */

#ifdef FIX_SIZE
#define DBG_FL(x) FX2FL(x)
#else
#define DBG_FL(x) x
#endif // FIX_SIZE

const char* FL_FMT = "%11.6f ";
const char* INT_FMT = "%d ";


#define PRINT_FL(txt, fl)          \
    printf("%s",txt);              \
    printf(FL_FMT,DBG_FL(fl))

#define PRINT_FL_LN(txt, fl)       \
    printf("%s",txt);              \
    printf(FL_FMT,DBG_FL(fl));     \
    printf("%s\n","")

#define PRINT_INT(txt, n)          \
    printf("%s", txt);             \
    printf(INT_FMT, n)

#define PRINT_INT_LN(txt, n)       \
    printf("%s", txt);             \
    printf(INT_FMT, n);            \
    printf("%s\n","")

#define PRINT_TXT_LN(txt)          \
    printf("%s", txt);             \
    printf("%s\n","")


