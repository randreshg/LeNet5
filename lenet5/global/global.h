#pragma once

/* ----- DATA TYPES ----- */
typedef enum { false, true } bool;
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
#include <time.h>

/* ----- ARRAY ----- */
struct array
{
    uint n;
    number *p;
};
#define ARRAY_VALUE(ar, n) *((ar)->p+n)
extern void ARRAY_FREE(Array **a);
extern Array *ARRAY(uint n);

/* ----- FEATURE ----- */
struct feature
{
    uint n;
    Matrix **matrix;
};
#define FEATURE_GETMATRIX(f, na) ((f)->matrix + na)
#define FEATURE_MALLOCMATRIX(f) (f)->matrix = malloc(sizeof(Matrix *)*((f)->n))
extern void FEATURE_FREE(Feature **w);
extern Feature *FEATURE(uint8 n, uint8 fl);


/* ----- MATRIX ----- */
struct matrix
{
    uint n, m;
    number *p;
};
#define MATRIX_VALUE(ma, ni, mi) *((ma)->p + ni*((ma)->m) + mi)
#define MATRIX_VALUE1(ma, ni) *((ma)->p + ni)
#define MATRIX_SIZE(ma) (((ma)->n)*((ma)->m))
extern void MATRIX_FREE(Matrix **a);
extern Matrix *MATRIX(number n, number m);

/* ----- WEIGHT ----- */
struct weight
{
    uint n, m;
    Matrix **matrix;
};

#define WEIGHT_GETMATRIX(w, ni, mi) ((w)->matrix + ni*((w)->m) + mi)
#define WEIGHT_GETMATRIX1(w, i) ((w)->matrix + i)
#define WEIGHT_MALLOCMATRIX(w) (w)->matrix = malloc(sizeof(Matrix *)*((w)->n)*((w)->m))
#define WEIGHT_SIZE(w) (((w)->n)*((w)->m))
extern void WEIGHT_FREE(Weight **w);
extern Weight *WEIGHT(uint n, uint m, uint wm_n, uint wm_m);

