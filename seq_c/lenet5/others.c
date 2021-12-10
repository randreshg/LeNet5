#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
number ReLU(number x) {
    return x*(x > 0);
}

number ReLU_GRAD(number x) {
    return x > 0;
}

void loadInput(uint8 *input, Feature *features) {
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(features, 0);
    uint in, im;
    number mean = 0, std = 0, val;
    //Calculate standard deviation and mean
    for(in = 0; in < IMG_SIZE; in++){
        val = input[in];
        mean += val;
        std += val*val;
    }
    mean = mean/IMG_SIZE;
    std = sqrt(std/IMG_SIZE - mean*mean);
    //Normalize data and add padding
    for(in = 0; in < IMG_ROWS; in++)
    for(im = 0; im < IMG_COLS; im++)
        MATRIX_VALUE(inputMatrix, (in + 2), (im + 2)) = (input[in*IMG_COLS + im] - mean) / std;
}

void softMax(Feature *input, uint8 target, Feature *featureGradient) {
    //Aux variables
    uint8 on, om;
    number den = 0, inner = 0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    Matrix *gradientMatrix = FEATURE_GETMATRIX(featureGradient, 0);
    //Error and softmax
    for(on = 0; on < inputMatrix->m; on++) {
        den = 0;
        for(om = 0; om < inputMatrix->m; om++)
            den += exp(MATRIX_VALUE1(inputMatrix, om) - MATRIX_VALUE1(inputMatrix, on));
        ARRAY_VALUE(gradientMatrix, on) = 1.0/den;
        inner -= ARRAY_VALUE(gradientMatrix, on) * ARRAY_VALUE(gradientMatrix, on);
    }
    inner += ARRAY_VALUE(gradientMatrix, target);
    for(om = 0; om < gradientMatrix->m; om++)
        ARRAY_VALUE(gradientMatrix, om) *= (om == target) - MATRIX_VALUE1(gradientMatrix, om) - inner;
}

number costFunction(Feature *input, uint8 target) {
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    return (-log(MATRIX_VALUE(FEATURE_GETMATRIX(input, 0), 0, target))/inputMatrix->m);
}
