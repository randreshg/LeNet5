#pragma once

#define MALLOC_FEATURE(f) f->pointer = malloc(sizeof(float)*f->length_filter*f->length_feature*f->length_feature)
//#define MALLOC_FEATURE(f) f.pointer = malloc(sizeof(float)*f.length_filter*f.length_feature*f.length_feature)
#define GET_LENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GET_COUNT(array)  (sizeof(array)/sizeof(double))

/* ----- FUNCTIONS ----- */
void convolute(double **input, double **weight, double **output );
void convolution(Feature *input, LeNet lenet);
void subsampling(double ***input, double ***output);
void initialValues(double ***data);
void dotproduct(double ***input, double **weight, double *bias, double *output);