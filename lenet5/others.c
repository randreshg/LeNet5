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
    uint8 om;
    number den = 0.0;
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    Matrix *gradientMatrix = FEATURE_GETMATRIX(featureGradient, 0);
    //Get denominator
    
    for(om = 0; om < inputMatrix->m; om++){
        printf("%f, ", MATRIX_VALUE1(inputMatrix, om));
        MATRIX_VALUE1(gradientMatrix, om) = exp(MATRIX_VALUE1(inputMatrix, om));
        den += MATRIX_VALUE1(gradientMatrix, om);
    }
    printf("\n");
    for(om = 0; om < inputMatrix->m; om++){
        //Softmax calculation
        ARRAY_VALUE(gradientMatrix, om) = MATRIX_VALUE1(gradientMatrix, om)/den;
        //printf("%f \n", MATRIX_VALUE1(gradientMatrix, om));
        //Softmax gradient
        if(om==target)
            MATRIX_VALUE1(gradientMatrix, om) = MATRIX_VALUE1(gradientMatrix,om)-1;
    }
}

number costFunction(Feature *input, uint8 target){
    //Aux variables
    Matrix *inputMatrix = FEATURE_GETMATRIX(input, 0);
    return (-log(MATRIX_VALUE(FEATURE_GETMATRIX(input, 0), 0, target))/inputMatrix->m);
}


