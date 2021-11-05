#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
void softMax(Feature *input){
    //Output malloc
    Array *output = ARRAY(OUTPUT);
    //Aux variables
    uint om;
    number den = 0.0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    //Get denominator
    for(om = 0; om < inputMatrix->m; om++){
        MATRIX_VALUE(inputMatrix, 0, om) = exp(MATRIX_VALUE(inputMatrix, 0, om));
        den += MATRIX_VALUE(inputMatrix, 0, om);
    }
    //Softmax calculation
    for(om = 0; om < inputMatrix->m; om++)
        ARRAY_VALUE(output, om) = MATRIX_VALUE(inputMatrix, 0, om)/den;
    //Softmax gradient
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
