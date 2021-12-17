/*
Takafumi Hoiruchi. 2018.
https://github.com/takafumihoriuchi/MNIST_for_C

Edited by: Rafael Herrera /2021
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

// set appropriate path for data
#define TRAIN_IMAGE "data/train-images-idx3-ubyte"
#define TRAIN_LABEL "data/train-labels-idx1-ubyte"
#define TEST_IMAGE "data/t10k-images-idx3-ubyte"
#define TEST_LABEL "data/t10k-labels-idx1-ubyte"

#define IMG_ROWS 28  // 28
#define IMG_COLS 28  // 28
#define IMG_SIZE 784 // IMG_ROWS*IMG_COLS
#define NUM_TRAIN 60000
#define NUM_TEST 10000

#define MAX_IMAGESIZE 1280
#define MAX_BRIGHTNESS 255
#define MAX_FILENAME 256
#define MAX_NUM_OF_IMAGES 1


extern void read_data(const int count, const char data_file[], const char label_file[], unsigned char data[][IMG_SIZE], unsigned char label[]);
extern void image_char2float(unsigned char data_image_char[IMG_SIZE], float data_image[IMG_SIZE]);
extern void images_char2float(int num_data, unsigned char data_image_char[][IMG_SIZE], float data_image[][IMG_SIZE]);
extern void load_mnist(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char train_label[NUM_TRAIN], unsigned char test_label[NUM_TEST]);
extern void load_testData(unsigned char test_image[NUM_TEST][IMG_SIZE], unsigned char test_label[NUM_TEST]);
extern void load_trainingData(unsigned char train_image[NUM_TRAIN][IMG_SIZE], unsigned char train_label[NUM_TRAIN]);
extern void print_mnist_img(float data_image[IMG_SIZE]);
extern void print_mnist_pixel(float data_image[][IMG_SIZE], int num_data);
extern void print_mnist_label(int data_label[], int num_data, int train_label[NUM_TRAIN], int test_label[NUM_TEST]);
