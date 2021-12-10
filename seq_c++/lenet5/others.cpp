#include "lenet.h"

/* ----- OTHERS FUNCTIONS ----- */
number ReLU(number x) {
    return x*(x > 0);
}

number ReLU_GRAD(number x) {
    return x > 0;
}

void loadInput(uint8 *input, number output[INPUT][LENGTH_FEATURE0][LENGTH_FEATURE0]) {
    //Aux variables
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
        output[0][in + 2][im + 2] = (input[in*IMG_COLS + im] - mean) / std;
}

void softMax(number input[OUTPUT], uint8 target, number outputGradient[OUTPUT]) {
    //Aux variables
    uint8 on, om;
    number den = 0, inner = 0;
    //Error and softmax
    for(on = 0; on < OUTPUT; on++) {
        den = 0;
        for(om = 0; om < OUTPUT; om++)
            den += exp(input[om] - input[on]);
        outputGradient[on] = 1.0/den;
        inner -= outputGradient[on] * outputGradient[on];
    }
    inner += outputGradient[target];
    for(on = 0; on < OUTPUT; on++)
        outputGradient[on] *= (on == target) - outputGradient[on] - inner;
}