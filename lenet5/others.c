#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
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

number costFunction(Feature *input, Array *target){
    //Aux variables
    uint om;
    number cost = 0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    for(om =0; om<inputMatrix->m; om++){
        cost = ARRAY_VALUE(target, om)*log(MATRIX_VALUE(inputMatrix, 0, om));
    }
    return (-cost/inputMatrix->m);
}


