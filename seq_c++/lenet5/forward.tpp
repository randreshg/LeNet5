#include "lenet.h"

/* ----- FORWARD FUNCTIONS ----- */
template<size_t ON, size_t ON1, size_t OM1, size_t BN>
void activation_forward(number (&output)[ON][ON1][OM1], number (&bias)[BN]) {
    uint on, om, matrixSize = ON1*OM1;
    for(on = 0; on < ON; on++) 
    for(om = 0; om < matrixSize; om++)
        ((number *)output[on])[om] = ReLU(((number *)output[on])[om] + bias[on]);
}

template<size_t IN, size_t IM, size_t WN, size_t WM, size_t ON, size_t OM>
void convolute_forward(number (&input)[IN][IM], number (&weight)[WN][WM], number (&output)[ON][OM]) {
    //Aux variables
    uint on, om, wn, wm;
    //Output loop
    for(on = 0; on < ON; on++)
    for(om = 0; om < OM; om++)
        //Weight matrix loop - KERNEL
        for(wn = 0; wn < WN; wn++)
        for(wm = 0; wm < WM; wm++)
            //Cross-Correlation
            output[wn][wm] += output[on + wn][om + wm] * weight[wn][wm];
}

template<size_t IN, size_t IN1, size_t IM1, size_t WN, size_t WM, size_t WN1, size_t WM1, size_t BN, size_t ON, size_t ON1, size_t OM1>
void convolution_forward(number (&input)[IN][IN1][IM1], number (&weight)[WN][WM][WN1][WM1],number (&bias)[BN], number (&output)[ON][ON1][OM1]) {
    //Aux variables
    uint wn, wm;
    //Convolution
    for(wn = 0; wn < WN; wn++)
    for(wm = 0; wm < WM; wm++)
        convolute_forward(input[wn], weight[wn][wm], output[wm]);
    //Activation function
    activation_forward(output, bias);
}

template<size_t IN, size_t IN1, size_t IM1, size_t ON, size_t ON1, size_t OM1>
void subsampling_forward(number (&input)[IN][IN1][IM1], number (&output)[ON][ON1][OM1]) {
    //Aux variables
    uint o, on, om, ln, lm, aux_n, aux_m;
    number max, aux;
    const uint lnLength = IN1/ON1, lmLength = IM1/OM1;
    //Ouput array loop
    for(o = 0; o < ON; o++)
        //Output matrix loop
        for(on = 0; on < ON1; on++)
        for(om = 0; om < OM1; om++) {
            //Subsampling
            max = -1, aux_n = lnLength*on, aux_m = lmLength*om;
            for(ln = 0; ln < lnLength; ln++)
            for(lm = 0; lm < lmLength; lm++) {
                aux = input[o][aux_n + ln][aux_m + lm];
                max = (aux > max) ? aux:max;
            }
            output[o][on][om] = max;
        }
}

template<size_t IN, size_t IN1, size_t IM1, size_t WN, size_t WM, size_t BN, size_t ON>
void dotproduct_forward(number (&input)[IN][IN1][IM1], number (&weight)[WN][WM], number (&bias)[BN], number (&output)[ON]) {
    //Aux variables
    uint wn, wm;
    //Dot product
    for(wn = 0; wn < WN; wn++) 
    for(wm = 0; wm < WM; wm++)
        output[wm] += ((number *)input)[wn] * weight[wn][wm];
    //Activation function
    for(wn = 0; wn < BN; wn++)
        output[wn] = ReLU(output[wn] + bias[wn]);
}
