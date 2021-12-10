MKDIR_P := mkdir -p
OUT_DIR := build
OBJDIR    =   objdir
OBJ = $(addprefix $(OBJDIR)/, $(patsubst %.cpp, %.o, $(wildcard *.cpp)))

lenet: main.o lenet.o backward.o forward.o others.o mnist.o
	g++ -o lenet $(OUT_DIR)/main.o $(OUT_DIR)/lenet.o $(OUT_DIR)/backward.o $(OUT_DIR)/forward.o $(OUT_DIR)/others.o $(OUT_DIR)/mnist.o -lm

main.o: main.cpp
	g++ -c main.cpp -o $(OUT_DIR)/$@

lenet.o: lenet5/lenet.cpp
	g++ -c lenet5/lenet.cpp -o $(OUT_DIR)/$@

backward.o: lenet5/backward.cpp
	g++ -c lenet5/backward.cpp -o $(OUT_DIR)/$@

forward.o: lenet5/forward.cpp
	g++ -c lenet5/forward.cpp -o $(OUT_DIR)/$@

others.o: lenet5/others.cpp
	g++ -c lenet5/others.cpp -o $(OUT_DIR)/$@

mnist.o: lenet5/mnist/mnist.cpp
	g++ -c lenet5/mnist/mnist.cpp -o $(OUT_DIR)/$@

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
