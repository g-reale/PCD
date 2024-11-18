COMPILER = gcc
CFLAGS = -g -fsanitize=address -Wall -Wextra
LIBS = -lm -lncurses

SRC_FILES = $(wildcard *.c)
OBJ_FILES = $(patsubst %.c, object/%.o, $(SRC_FILES))
TARGET = main

all: $(TARGET)

$(TARGET): $(OBJ_FILES)
	$(COMPILER) $(CFLAGS) -o $@ $^ $(LIBS)

object/%.o: %.c
	@mkdir -p object
	$(COMPILER) $(CFLAGS) -c $< -o $@

clean:
	rm -rf object $(TARGET)

.PHONY: all clean
