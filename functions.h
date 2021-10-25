#pragma once

#define GET_LENGTH(array) (sizeof(array)/sizeof(*(array)))

/* ----- FUNCTIONS ----- */
void convolute(float **input, float **weight, float **output );
void convolution(float ***input, float ****weight, float *bias, float ***output);
void subsampling(float ***input, float ***output);
void initialValues(float ***data);