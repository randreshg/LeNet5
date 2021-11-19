#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
number ReLU(number x){
    return x*(x > 0);
}

number ReLU_GRAD(number y){
    return y > 0;
}

void softMax(Feature *input, uint8 target, Feature *featureGradient){
    //Aux variables
    uint on, om;
    number den, inner = 0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    Matrix *gradientMatrix = FEATURE_GETMATRIX(featureGradient, 0);
    //Error and softmax
    for(on = 0; on < inputMatrix->m; on++){
        //printf("%2f, ", MATRIX_VALUE1(inputMatrix, on));
        den = 0;
        for(om = 0; om < inputMatrix->m; om++)
            den += exp(MATRIX_VALUE1(inputMatrix, om) - MATRIX_VALUE1(inputMatrix, on));
        ARRAY_VALUE(gradientMatrix, on) = 1. / den;
        inner -= ARRAY_VALUE(gradientMatrix, on) * ARRAY_VALUE(gradientMatrix, on);
    }
    //printf("\n");
    inner += ARRAY_VALUE(gradientMatrix, target);
    for(om = 0; om < inputMatrix->m; om++)
        ARRAY_VALUE(gradientMatrix, om) *= (om == target) - MATRIX_VALUE1(gradientMatrix, om) - inner;
}

number costFunction(Feature *input, uint8 target){
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    return (-log(MATRIX_VALUE(FEATURE_GETMATRIX(input, 0), 0, target))/inputMatrix->m);
}


