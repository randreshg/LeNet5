MKDIR_P := mkdir -p
OUT_DIR := build
OBJDIR    =   objdir
OBJ = $(addprefix $(OBJDIR)/, $(patsubst %.c, %.o, $(wildcard *.c)))

lenet: main.o lenet.o backward.o forward.o others.o global.o mnist.o
	gcc -o lenet $(OUT_DIR)/main.o $(OUT_DIR)/lenet.o $(OUT_DIR)/backward.o $(OUT_DIR)/forward.o $(OUT_DIR)/others.o $(OUT_DIR)/global.o $(OUT_DIR)/mnist.o -lm -fopenmp

lenet_openmp: main_openmp.o lenet_openmp.o backward.o forward.o others.o global.o mnist.o
	gcc -o lenet_openmp $(OUT_DIR)/main_openmp.o $(OUT_DIR)/lenet_openmp.o $(OUT_DIR)/backward.o $(OUT_DIR)/forward.o $(OUT_DIR)/others.o $(OUT_DIR)/global.o $(OUT_DIR)/mnist.o -lm -fopenmp

main.o: main.c
	gcc -c main.c -o $(OUT_DIR)/$@ -fopenmp -DOPENMP=0

main_openmp.o: main.c
	gcc -c main.c -o $(OUT_DIR)/$@ -fopenmp -DOPENMP=1

lenet.o: lenet5/lenet.c
	gcc -c lenet5/lenet.c -o $(OUT_DIR)/$@ -DOPENMP=0

lenet_openmp.o: lenet5/lenet.c
	gcc -c lenet5/lenet.c -o $(OUT_DIR)/$@ -fopenmp -DOPENMP=1

backward.o: lenet5/backward.c
	gcc -c lenet5/backward.c -o $(OUT_DIR)/$@

forward.o: lenet5/forward.c
	gcc -c lenet5/forward.c -o $(OUT_DIR)/$@

others.o: lenet5/others.c
	gcc -c lenet5/others.c -o $(OUT_DIR)/$@

global.o: lenet5/global/global.c
	gcc -c lenet5/global/global.c -o $(OUT_DIR)/$@

mnist.o: lenet5/mnist/mnist.c
	gcc -c lenet5/mnist/mnist.c -o $(OUT_DIR)/$@

# Aux
.PHONY: directories all clean

all: $(OUT_DIR)/program
all_openmp: $(OUT_DIR)/program_openmp

directories: $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)
	rm ./lenet

$(OUT_DIR):
	${MKDIR_P} $(OUT_DIR)

$(OUT_DIR)/program: | directories
	make lenet

$(OUT_DIR)/program_openmp: | directories
	make lenet_openmp