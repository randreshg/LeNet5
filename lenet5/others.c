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
    for(om = 0; om < inputMatrix->m; om++)
        den += MATRIX_VALUE(inputMatrix, 0, om);
    //Softmax calculation
    for(om = 0; om < inputMatrix->m; om++)
        ARRAY_VALUE(output, om) = MATRIX_VALUE(inputMatrix, 0, om)/den;
    //Softmax gradient


}

// void costFunction(Feature *input, Array *target){
//     uint m = target->n;
// }
