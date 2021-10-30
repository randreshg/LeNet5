#pragma once

/* ----- DATA TYPES ----- */
typedef unsigned char uint8;
typedef unsigned int uint;
typedef uint8 image[28][28];
typedef float number;

struct Feature;
struct Matrix;
struct Weight;
struct Array;

/* ----- DEPENDENCIES ----- */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/* ----- OTHER ----- */
#include "Array.h"
#include "Weight.h"
#include "Feature.h"
#include "Matrix.h"