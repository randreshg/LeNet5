#pragma once
#include "global.h"

/* ----- FEATURE ----- */
typedef struct Feature
{
    uint n;
    Matrix *matrix;
} Feature;

#define FEATURE_GETMATRIX(f, na) (f->matrix + na)
#define FEATURE_MALLOCMATRIX(f) f->matrix = malloc(sizeof(Matrix)*(f->n))
#define FEATURE_FREEMATRIX(w) free(w->matrix)


Feature *FEATURE(uint8 n, uint8 fl)
{
    Feature *fe = (Feature *)malloc(sizeof(Feature));
    fe->n = n, fe->matrix = NULL;
    return fe;
}
