#pragma once
#include <stdlib.h>


/* ----- DATA TYPES ----- */
typedef unsigned char uint8;
typedef unsigned int uint;
typedef uint8 image[28][28];
typedef float number;

struct Feature;
struct Matrix;
struct Weight;
struct Array;

#include "Array.h"
#include "Weight.h"
#include "Feature.h"
#include "Matrix.h"