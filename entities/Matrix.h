#pragma once
#include "global.h"

/* ----- MATRIX ----- */
typedef struct Matrix
{
    uint8 n, m;
    number *p;
} Matrix;


Matrix *MATRIX(uint8 n, uint8 m)
{
    Matrix *ma = (Matrix *)malloc(sizeof(Matrix));
    ma->n = n, ma->m = m;
    ma->p = (number *)malloc(sizeof(number)*(n)*(m));
    return ma;
}
