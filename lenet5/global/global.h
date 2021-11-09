#pragma once

/* ----- DATA TYPES ----- */
typedef unsigned char uint8;
typedef unsigned int uint;
typedef uint8 image[28][28];
typedef float number;

typedef struct feature Feature;
typedef struct matrix Matrix;
typedef struct weight Weight;
typedef struct array Array;

/* ----- DEPENDENCIES ----- */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ----- ARRAY ----- */
struct array
{
    uint n;
    number *p;
};
#define ARRAY_VALUE(ar, n) *(ar->p+n)
extern Array *ARRAY(uint n);

/* ----- FEATURE ----- */
struct feature
{
    uint n;
    Matrix *matrix;
};
#define FEATURE_GETMATRIX(f, na) (f->matrix + na)
#define FEATURE_MALLOCMATRIX(f) f->matrix = malloc(sizeof(Matrix)*(f->n))
#define FEATURE_FREEMATRIX(w) free(w->matrix)
extern Feature *FEATURE(uint8 n, uint8 fl);

/* ----- MATRIX ----- */
struct matrix
{
    uint n, m;
    number *p;
};
#define MATRIX_VALUE(ma, ni, mi) *(ma->p + ni*ma->m + mi)
#define MATRIX_VALUE1(ma, ni) *(ma->p + ni)
#define MATRIX_SIZE(ma) ma->n*ma->m
extern Matrix *MATRIX(number n, number m);

/* ----- WEIGHT ----- */
struct weight
{
    uint n, m;
    Matrix *matrix;
};
#define WEIGHT_GETMATRIX(w, na, ma) w->matrix + na*w->m + ma
#define WEIGHT_MALLOCMATRIX(w) w->matrix = malloc(sizeof(Matrix)*(w->n)*(w->m))
#define WEIGHT_FREEMATRIX(w) free(w->matrix)
extern Weight *WEIGHT(uint n, uint m);
