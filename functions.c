#include "functions.h"
#include <stdlib.h>

/* ----- FUNCTIONS ----- */
void convolution(float ***input, float ***weight, float *bias, float ***output){
    int i, j;
    for(i=0; i<100; i++){
        for(j=0; j<100; j++){




        }
    }
}

#define ReLU(x) (x>0? x: 0)

/**********************************************************************/
float TanH(float x){
	return 2.0/(1.0+exp(-2*x))-1.0;
}