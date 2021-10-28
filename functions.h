#pragma once

/* ----- FUNCTIONS ----- */
void convolute(Matrix *input, Matrix *weight, Array *bias , Matrix *output );
void convolution(Feature *input, LeNet lenet);
void subsampling(Feature *input);
void initialValues(double ***data);
void dotproduct(double ***input, double **weight, double *bias, double *output);