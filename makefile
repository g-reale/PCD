COMPILER = nvcc
CFLAGS = -Xcompiler "-g -Wall -Wextra -fopenmp"
LIBS = -lm -lncurses

SRC_FILES = $(wildcard *.c)
CU_SRC_FILES = $(wildcard *.cu)
C_OBJ_FILES = $(patsubst %.c, object/%.o, $(SRC_FILES))
CU_OBJ_FILES = $(patsubst %.cu, object/%.o, $(CU_SRC_FILES))
OBJ_FILES = $(C_OBJ_FILES) $(CU_OBJ_FILES)
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(COMPILER) $(CFLAGS) -o $@ $^ $(LIBS)

object/%.o: %.c
	@mkdir -p object
	$(COMPILER) $(CFLAGS) -c $< -o $@

object/%.o: %.cu
	@mkdir -p object
	$(COMPILER) $(CFLAGS) -c $< -o $@

clean:
	rm -rf object $(TARGET)

run: all
	./main || reset

debug: all
	./main 2> stderr.log || reset && cat stderr.log

update:
	rclone sync . gdrive: -P --filter-from filter.txt

.PHONY: all clean run debug update