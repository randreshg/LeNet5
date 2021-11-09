#include "global.h"

Array *ARRAY(uint n){
    Array *ar = (Array *)malloc(sizeof(Array));
    ar->n = n, ar->p = calloc(n, sizeof(number));
    return ar;
}

Feature *FEATURE(uint8 n, uint8 fl){
    Feature *fe = (Feature *)malloc(sizeof(Feature));
    fe->n = n, fe->matrix = NULL;
    return fe;
}

Matrix *MATRIX(number n, number m){
    Matrix *ma = (Matrix *)malloc(sizeof(Matrix));
    ma->n = n, ma->m = m;
    ma->p = calloc(n*m, sizeof(number));
    return ma;
}

Weight *WEIGHT(uint n, uint m){
    Weight *we = (Weight *)malloc(sizeof(Weight));
    we->n = n, we->m = m, we->matrix = NULL;
    return we;
}