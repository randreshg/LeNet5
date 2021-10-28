#pragma once
#include "global.h"

/* ----- ARRAY ----- */
typedef struct Array
{
    uint n;
    number *p;
} Array;

#define ARRAY_VALUE(ar, n) *(ar->p+n)

Array *ARRAY(uint n)
{
    Array *ar = (Array *)malloc(sizeof(Array));
    ar->n = n, ar->p = (number *)malloc(sizeof(number)*(n));
    return ar;
}
