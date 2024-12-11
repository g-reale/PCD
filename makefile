COMPILER = nvcc
CFLAGS = -Xcompiler "-g -Wall -Wextra -fopenmp" 
LIBS = -Xlinker "-lm -lncurses"

SRC_FILES = $(wildcard *.c)
CU_SRC_FILES = $(wildcard *.cu)
OBJ_FILES = $(patsubst %.c, object/%.o, $(SRC_FILES))
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ_FILES) $(CU_SRC_FILES)
	$(COMPILER) $(CFLAGS) -o $@ $^ $(LIBS)

object/%.o: %.c
	@mkdir -p object
	$(COMPILER) $(CFLAGS) -c $< -o $@

clean:
	rm -rf object $(TARGET)

run: all
	./main || reset

debug: all
	./main 2> stderr.log || reset && cat stderr.log

.PHONY: all clean run debug 
