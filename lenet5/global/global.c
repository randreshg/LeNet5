#include "global.h"

/* ----- ARRAY ----- */
Array *ARRAY(uint n){
    Array *ar = (Array *)malloc(sizeof(Array));
    ar->n = n, ar->p = calloc(n, sizeof(number));
    return ar;
}

void ARRAY_FREE(Array **ar){
    free((*ar)->p);
    free(*ar);
    *ar = NULL;
} 

/* ----- FEATURE ----- */
Feature *FEATURE(uint n, uint fm_n, uint fm_m){
    Feature *fe = (Feature *)malloc(sizeof(Feature));
    fe->n = n;
    //Malloc matrix
    FEATURE_MALLOCMATRIX(fe);
    uint i;
    for(i=0; i<fe->n; i++)
        fe->matrix[i] = MATRIX(fm_n, fm_m);
    return fe;
}

void FEATURE_FREE(Feature **fe){
    uint i;
    for(i=0; i<(*fe)->n; i++)
        MATRIX_FREE(FEATURE_GETMATRIXP(*fe, i));
    free((*fe)->matrix);
    free(*fe);
    *fe = NULL;
} 

/* ----- MATRIX ----- */
Matrix *MATRIX(number n, number m){
    Matrix *ma = (Matrix *)malloc(sizeof(Matrix));
    ma->n = n, ma->m = m;
    ma->p = calloc(n*m, sizeof(number));
    return ma;
}

void MATRIX_FREE(Matrix **ma){
    free((*ma)->p);
    free(*ma);
    *ma = NULL;
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

void WEIGHT_FREE(Weight **we){
    uint i, size = WEIGHT_SIZE(*we);
    printf("--------\nFREE MATRIX %u\n", size);
    for(i=0; i<size; i++)
        MATRIX_FREE(WEIGHT_GETMATRIX1P(*we, i));
    printf("FREE POINTER \n");
    free((*we)->matrix);
    free(*we);
    *we = NULL;
} 