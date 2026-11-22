.PHONY: all tcp shm shmr shms clean 

all: tcp shm 

tcp: 
	@mkdir -p build 
	go build -o build/gpuconbench ./cmd/demo

shm: 
	@mkdir -p build 
	go build -o build/shmbench ./cmd/shmbench

shmr:
	./build/shmbench receiver
	
shms:
	./build/shmbench sender

clean:
	rm -rf build