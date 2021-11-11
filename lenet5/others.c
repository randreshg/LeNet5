#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
number ReLU(number x){
    return x*(x > 0);
}

number ReLU_GRAD(number y){
    return y > 0;
}

void softMax(Feature *input, Array *target, Feature *featureGradient){
    //Aux variables
    uint om;
    number den = 0.0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    Matrix *gradientMatrix = FEATURE_GETMATRIX(featureGradient, 0);
    //Get denominator
    for(om = 0; om < inputMatrix->m; om++){
        MATRIX_VALUE(inputMatrix, 0, om) = exp(MATRIX_VALUE(inputMatrix, 0, om));
        den += MATRIX_VALUE(inputMatrix, 0, om);
    }
    for(om = 0; om < inputMatrix->m; om++){
        //Softmax calculation
        ARRAY_VALUE(inputMatrix, om) = MATRIX_VALUE(inputMatrix, 0, om)/den;
        //Softmax gradient
        MATRIX_VALUE(gradientMatrix, 0, om) = MATRIX_VALUE(inputMatrix, 0, om) - ARRAY_VALUE(target, om);
    }
}

number costFunction(Feature *input, uint8 target){
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    return (-log(MATRIX_VALUE(FEATURE_GETMATRIX(input, 0), 0, target))/inputMatrix->m);
}

