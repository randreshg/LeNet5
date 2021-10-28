#pragma once
#include "global.h"

/* ----- WEIGHT ----- */
typedef struct Weight
{
    uint n, m;
    Matrix *matrix;
} Weight;

#define WEIGHT_GETMATRIX(w, na, ma) w->matrix + na*w->m + ma
#define WEIGHT_MALLOCMATRIX(w) w->matrix = malloc(sizeof(Matrix)*(w->n)*(w->m))
#define WEIGHT_FREEMATRIX(w) free(w->matrix)


Weight *WEIGHT(uint8 n, uint8 m)
{
    Weight *we = (Weight *)malloc(sizeof(Weight));
    we->n = n, we->m = m, we->matrix = NULL;
    return we;
}
