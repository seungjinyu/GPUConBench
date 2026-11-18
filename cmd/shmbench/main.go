package main

import (
	"fmt"
	"os"
	"syscall"
)

const shmPath = "/dev/shm/gpuconbench-shm-test"

func createSharedMemory(path string, size int)(*os.File, []byte, error){

	f, err := os.OpenFile(
		path, 
		os.O_CREATE|os.O_EXCL|os.O_RDWR,
		0600,
	)

	if err != nil {
		return nil, nil, err
	}

	// Change the file size
	if err := f.Truncate(int64(size)); err != nil{
		f.Close()
		os.Remove(path)
		return nil, nil, err 
	}

	// Map the file into process memory
	data, err := syscall.Mmap(
		int(f.Fd()),
		0,
		size,
		syscall.PROT_READ | syscall.PROT_WRITE ,
		syscall.MAP_SHARED,
	)

	if err != nil {
		f.Close()
		os.Remove(path)
		return nil, nil, err 
	}


	return f, data, nil
}

func main() {
	fmt.Println("GPUConBench Shared Memory Benchmark")

	size := 64*1024

	f , shared, err := createSharedMemory(shmPath, size)

	if err != nil {
		fmt.Println("Error: ", err)
	}

	defer os.Remove(shmPath)
	defer f.Close()
	defer syscall.Munmap(shared)

	fmt.Println("Shared memory created")
	fmt.Println("Size: ",len(shared),"bytes")

	fmt.Println("Press Enter to exit...")
	fmt.Scanln()

	
}
