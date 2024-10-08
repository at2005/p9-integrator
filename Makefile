# for h100s, for other GPUs replace sm_90 with appropriate arch
# mint == mercury/mercurius integrator
CC=nvcc
CFLAGS=-arch=sm_90 -O3
BIN_DIR=bin
TARGET=$(BIN_DIR)/mint
SOURCE=src/mercurius.cu

.PHONY: all clean

all: $(TARGET)

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(TARGET): $(SOURCE) | $(BIN_DIR)
	$(CC) $(SOURCE) -o $(TARGET) $(CFLAGS)

clean:
	rm -rf $(BIN_DIR)
	for i in $(shell seq 0 7); do rm -f output_$$i; rm -f output_$$i.gz; done