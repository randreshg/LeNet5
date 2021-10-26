#pragma once

#define GET_LENGTH(array) (sizeof(array)/sizeof(*(array)))
#define GET_COUNT(array)  (sizeof(array)/sizeof(double))
/* ----- FUNCTIONS ----- */
void convolute(double **input, double **weight, double **output );
void convolution(double ***input, double ****weight, double *bias, double ***output);
void subsampling(double ***input, double ***output);
void initialValues(double ***data);
void dotproduct(double ***input, double **weight, double *bias, double *output);