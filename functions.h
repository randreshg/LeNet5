#pragma once

/* ----- FUNCTIONS ----- */
void convolution(float ***data);
void subsampling(float ***data);
void training(float ***data);
int predict(float ***data, int input);
void initialValues(float ***data);