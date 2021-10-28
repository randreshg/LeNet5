#pragma once
#include "global.h"

/* ----- FEATURE ----- */
typedef struct Feature
{
    uint8 n;
    Matrix *p;
} Feature;

#define FEATURE_MATRIX(w) w->p = malloc(sizeof(Matrix)*(w->n)*(w->m))

Feature *FEATURE(uint8 n, uint8 fl)
{
    Feature *fe = (Feature *)malloc(sizeof(Feature));
    fe->n = n, fe->p = NULL;
    return fe;
}
