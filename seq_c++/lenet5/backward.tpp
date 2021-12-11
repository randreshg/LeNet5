#include "lenet.h"

/* ----- BACKWARD FUNCTIONS ----- */
template<size_t IN, size_t IN1, size_t IM1, size_t ON, size_t ON1, size_t OM1>
void activation_backward(number (&input)[IN][IN1][IM1], number (&output)[ON][ON1][OM1]) {
    uint on, matrixSize = IN*IN1*IM1;
    for(on = 0; on < matrixSize; on++) 
        ((number *)output)[on] *= ReLU_GRAD(((number *)input)[on]);
}

template<size_t IN, size_t IM, size_t WN, size_t WM, size_t ON, size_t OM>
void convolute_backward(number (&input)[IN][IM], number (&weight)[WN][WM], number (&output)[ON][OM]) {
    //Aux variables
    uint in, im, wn, wm;
    //Input loop
    for(in = 0; in < IN; in++)
    for(im = 0; im < IM; im++)
        //Weight matrix loop
        for(wn = 0; wn < WN; wn++)
        for(wm = 0; wm < WM; wm++)
            //Cross-Correlation
            output[in + wn][im + wm] += input[in][im] * weight[wn][wm];
}

template<size_t IN, size_t IN1, size_t IM1, size_t IGN, size_t IGN1, size_t IGM1,
         size_t WN, size_t WM,  size_t WN1, size_t WM1, size_t WGN, size_t WGM, size_t WGN1, size_t WGM1, size_t BG1,
         size_t OGN, size_t OGN1, size_t OGM1>
void convolution_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN][IGN1][IGM1],
                          number (&weight)[WN][WM][WN1][WM1], number (&weightGradient)[WGN][WGM][WGN1][WGM1], number (&biasGradient)[BG1],
                          number (&outputGradient)[OGN][OGN1][OGM1]) {
    //Aux variables
    uint wn, wm, matrixSize;
    //Calculate output gradient
    for(wn = 0; wn < WN; wn++)
    for(wm = 0; wm < WM; wm++)
        convolute_backward(inputGradient[wm], weight[wn][wm], outputGradient[wn]);
    //Activation function
    activation_backward(input, outputGradient);
    //Update bias
    matrixSize = IGN1*IGM1;
    for(wn = 0; wn < IGN; wn++) 
    for(wm = 0; wm < matrixSize; wm++)
        biasGradient[wn] += ((number *)inputGradient[wn])[wm];
    //Update weights
    for(wn = 0; wn < WN; wn++)
    for(wm = 0; wm < WM; wm++)
        convolute_forward(input[wn], inputGradient[wm], weightGradient[wn][wm]);
}

template<size_t IN, size_t IN1, size_t IM1, size_t IGN, size_t IGN1, size_t IGM1, size_t OGN, size_t OGN1, size_t OGM1>
void subsampling_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN][IGN1][IGM1], number (&outputGradient)[OGN][OGN1][OGM1]) {
    //Aux variables
    uint i, in, im, ln, lm, maxLn, maxLm, aux_n, aux_m;
    number max, aux;
    const uint lnLength = OGN1/IGN1, lmLength = OGM1/IGM1;
    //Input array loop
    for(i = 0; i < IGN; i++) {
        //Input matrix loop
        for(in = 0; in < IGN1; in++)
        for(im = 0; im < IGM1; im++){
            //Subsampling
            max = -1.0, aux_n = lnLength*in, aux_m = lmLength*im;
            for(ln = 0; ln < lnLength; ln++) 
            for(lm = 0; lm < lmLength; lm++) {
                aux = input[i][aux_n + ln][aux_m + lm];
                if(aux > max)
                    max = aux, maxLn = (aux_n + ln), maxLm = (aux_m + lm);
            }
            outputGradient[i][maxLn][maxLm] = inputGradient[i][in][im];
        }
    }
}

template<size_t IN, size_t IN1, size_t IM1, size_t IGN,
         size_t WN, size_t WM, size_t WGN, size_t WGM, size_t BG1,
         size_t OGN, size_t OGN1, size_t OGM1>
void dotproduct_backward(number (&input)[IN][IN1][IM1], number (&inputGradient)[IGN],
                         number (&weight)[WN][WM], number (&weightGradient)[WGN][WGM], number (&biasGradient)[BG1],
                         number (&outputGradient)[OGN][OGN1][OGM1]) {
    //Aux variables
    uint wn, wm;
    //Dot product
    for(wn = 0; wn < WN; wn++) 
    for(wm = 0; wm < WM; wm++)
        ((number *)outputGradient)[wn] += inputGradient[wm] * weight[wn][wm];
    //Activation function
    activation_backward(input, outputGradient);
    //Update bias
    for(wn = 0; wn < IGN; wn++)
        biasGradient[wn] += inputGradient[wn];
    //Update weights
    for(wn = 0; wn < WN; wn++) 
    for(wm = 0; wm < WM; wm++)
        weightGradient[wn][wm] += ((number *)input)[wn] * inputGradient[wm];
}