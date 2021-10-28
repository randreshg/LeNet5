#pragma once

/* ----- FUNCTIONS ----- */
void convolute(double **input, double **weight, double **output );
void convolution(Feature *input, LeNet lenet);
void subsampling(double ***input, double ***output);
void initialValues(double ***data);
void dotproduct(double ***input, double **weight, double *bias, double *output);