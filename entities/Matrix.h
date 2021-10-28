#pragma once
#include "global.h"

/* ----- MATRIX ----- */
typedef struct Matrix
{
    uint n, m;
    number *p;
} Matrix;

#define MATRIX_VALUE(ma, ni, mi) *(ma->p + ni*ma->m + mi)

Matrix *MATRIX(uint8 n, uint8 m)
{
    Matrix *ma = (Matrix *)malloc(sizeof(Matrix));
    ma->n = n, ma->m = m;
    ma->p = (number *)malloc(sizeof(number)*(n)*(m));
    return ma;
}
