#pragma once

/* ----- DATA TYPES ----- */
typedef enum { false, true } bool;
typedef unsigned char uint8;
typedef unsigned int uint;
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
extern void ARRAY_FREE(Array **ar);
extern Array *ARRAY(uint n);

/* ----- FEATURE ----- */
struct feature
{
    uint n;
    Matrix **matrix;
};
#define FEATURE_GETMATRIX(fe, na) (*((fe)->matrix + na))
#define FEATURE_MALLOCMATRIX(fe) (fe)->matrix = malloc(sizeof(Matrix *)*((fe)->n))
extern void FEATURE_FREE(Feature **fe);
extern Feature *FEATURE(uint n, uint fm_n, uint fm_m);


/* ----- MATRIX ----- */
struct matrix
{
    uint n, m;
    number *p;
};
#define MATRIX_VALUE(ma, ni, mi) (*((ma)->p + ni*((ma)->m) + mi))
#define MATRIX_VALUE1(ma, ni) (*((ma)->p + ni))
#define MATRIX_SIZE(ma) (((ma)->n)*((ma)->m))
extern void MATRIX_FREE(Matrix **a);
extern Matrix *MATRIX(number n, number m);

/* ----- WEIGHT ----- */
struct weight
{
    uint n, m;
    Matrix **matrix;
};

#define WEIGHT_GETMATRIX(we, ni, mi) (*((we)->matrix + ni*((we)->m) + mi))
#define WEIGHT_GETMATRIX1(we, i) (*((we)->matrix + i))
#define WEIGHT_MALLOCMATRIX(we) (we)->matrix = malloc(sizeof(Matrix *)*((we)->n)*((we)->m))
#define WEIGHT_SIZE(we) (((we)->n)*((we)->m))
extern void WEIGHT_FREE(Weight **we);
extern Weight *WEIGHT(uint n, uint m, uint wm_n, uint wm_m);

