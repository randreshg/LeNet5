#pragma once
#include "global.h"

/* ----- ARRAY ----- */
typedef struct Array
{
    uint8 n;
    number *p;
} Array;


Array *ARRAY(uint n)
{
    Array *ar = (Array *)malloc(sizeof(Array));
    ar->n = n, ar->p = (number *)malloc(sizeof(number)*(n));
    return ar;
}
