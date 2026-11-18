.PHONY: all tcp shm clean 

all: tcp shm 

tcp: 
	@mkdir -p build 
	go build -o build/gpuconbench ./cmd/demo

shm: 
	@mkdir -p build 
	go build -o build/shmbench ./cmd/shmbench

clean:
	rm -rf build