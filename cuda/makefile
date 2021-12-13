MKDIR_P := mkdir -p
OUT_DIR := build
OBJDIR    =   objdir
OBJ = $(addprefix $(OBJDIR)/, $(patsubst %.cu, %.o, $(wildcard *.cu)))

lenet: main.o lenet.o others.o mnist.o
	nvcc -arch=sm_35 -rdc=true -o lenet $(OUT_DIR)/main.o $(OUT_DIR)/lenet.o $(OUT_DIR)/others.o $(OUT_DIR)/mnist.o -lm  

main.o: main.cu
	nvcc -arch=sm_35 -rdc=true -c main.cu -o $(OUT_DIR)/$@

lenet.o: lenet5/lenet.cu
	nvcc -arch=sm_35 -rdc=true -c lenet5/lenet.cu -o $(OUT_DIR)/$@

others.o: lenet5/others.cu
	nvcc -arch=sm_35 -rdc=true -dc -c lenet5/others.cu -o $(OUT_DIR)/$@

mnist.o: lenet5/mnist/mnist.cu
	nvcc -arch=sm_35 -rdc=true -c lenet5/mnist/mnist.cu -o $(OUT_DIR)/$@

# Aux
.PHONY: directories all clean

all: $(OUT_DIR)/program

directories: $(OUT_DIR)

clean:
	rm -rf $(OUT_DIR)
	rm ./lenet

$(OUT_DIR):
	${MKDIR_P} $(OUT_DIR)

$(OUT_DIR)/program: | directories
	make lenet
