SRC = $(wildcard src/*.c)

.PHONY: all clean

# CFLAGS are for the compiler
CFLAGS := -Wall -Wextra -g

# LDLIBS are for the linker (libraries)
LDLIBS := -lm

EXTRA_DEBUG_FLAGS := -fcolor-diagnostics -fansi-escape-codes
CC := cc

all: main

main: $(SRC)
	$(CC) $(CFLAGS) $^ -o $@ $(LDLIBS)

vscode-debug: $(SRC)
	$(CC) $(CFLAGS) $(EXTRA_DEBUG_FLAGS) $^ -o $@ $(LDLIBS)

lib: $(SRC)
	$(CC) $(CFLAGS) -fPIC -shared $^ -o tbtc.so $(LDLIBS)

clean:
	rm -rf *.o *~ main proxy.so
