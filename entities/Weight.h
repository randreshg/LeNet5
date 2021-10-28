#pragma once
#include "global.h"

/* ----- WEIGHT ----- */
typedef struct
{
    uint8 n, m;
    Matrix *p;
} Weight;

#define WEIGHT_MATRIX(w) w->p = malloc(sizeof(Matrix)*(w->n)*(w->m))

Weight *WEIGHT(uint8 n, uint8 m)
{
    Weight *we = (Weight *)malloc(sizeof(Weight));
    we->n = n, we->m = m, we->p = NULL;
    return we;
}
