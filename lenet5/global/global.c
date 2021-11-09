#include "global.h"

Array *ARRAY(uint n){
    Array *ar = (Array *)malloc(sizeof(Array));
    ar->n = n, ar->p = calloc(n, sizeof(number));
    return ar;
}

/* ----- FEATURE ----- */
Feature *FEATURE(uint8 n, uint8 fl){
    Feature *fe = (Feature *)malloc(sizeof(Feature));
    fe->n = n, fe->matrix = NULL;
    return fe;
}

void FEATURE_FREE(Feature *f){
    uint i;
    for(i=0; i<f->n; i++)
        MATRIX_FREE(*FEATURE_GETMATRIX(f, i));
    free(f->matrix);
    free(f);
} 

/* ----- MATRIX ----- */
Matrix *MATRIX(number n, number m){
    Matrix *ma = (Matrix *)malloc(sizeof(Matrix));
    ma->n = n, ma->m = m;
    ma->p = calloc(n*m, sizeof(number));
    return ma;
}

/* ----- WEIGHT ----- */
Weight *WEIGHT(uint n, uint m, uint wm_n, uint wm_m){
    Weight *we = (Weight *)malloc(sizeof(Weight));
    we->n = n, we->m = m;
    //Malloc matrix
    WEIGHT_MALLOCMATRIX(we);
    uint i, size = n*m;
    for(i=0; i<size; i++)
        we->matrix[i] = MATRIX(wm_n, wm_m);
    return we;
}

void WEIGHT_FREE(Weight *w){
    uint i, size = w->n*w->m;
    for(i=0; i<size; i++)
        MATRIX_FREE(*WEIGHT_GETMATRIX1(w, i));
    free(w->matrix);
    free(w);
} 