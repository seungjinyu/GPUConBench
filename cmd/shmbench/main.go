package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"os"
	"syscall"
	"time"
)

const shmPath = "/dev/shm/gpuconbench-shm-test"

// Create shared memory and map the file into the process memory
func createSharedMemory(path string, size int) (*os.File, []byte, error) {

	f, err := os.OpenFile(
		path,
		os.O_CREATE|os.O_EXCL|os.O_RDWR,
		0600,
	)

	if err != nil {
		return nil, nil, err
	}

	// Change the file size
	if err := f.Truncate(int64(size)); err != nil {
		f.Close()
		os.Remove(path)
		return nil, nil, err
	}

	// Map the file into process memory
	data, err := syscall.Mmap(
		int(f.Fd()),
		0,
		size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED,
	)

	if err != nil {
		f.Close()
		os.Remove(path)
		return nil, nil, err
	}

	return f, data, nil
}

func openSharedMemory(path string, size int) (*os.File, []byte, error) {

	f, err := os.OpenFile(path, os.O_RDWR, 0)
	if err != nil {
		return nil, nil, err
	}

	data, err := syscall.Mmap(
		int(f.Fd()),
		0,
		size,
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED,
	)

	if err != nil {
		f.Close()
		return nil, nil, err
	}

	return f, data, nil

}

func main() {
	fmt.Println("GPUConBench Shared Memory Benchmark")

	size := 64 * 1024

	if len(os.Args) < 2 {
		fmt.Println("Usage: shmbench receiver|sender")
		return
	}

	switch os.Args[1] {

	// receiver part
	case "receiver":
		f, shared, err := createSharedMemory(shmPath, size)
		if err != nil {
			fmt.Println("Error: ", err)
			return
		}

		defer os.Remove(shmPath)
		defer f.Close()
		defer syscall.Munmap(shared)

		fmt.Println("Receiver: Shared memory created")
		fmt.Println("Initial Value:", shared[0])

		socketPath := "/tmp/gpuconbench-shm-test.sock"

		listener, err := net.Listen("unix", socketPath)
		if err != nil {
			fmt.Println("Listen error: ", err)
			return
		}

		defer listener.Close()

		fmt.Println("Waiting for READY signal...")
		conn, err := listener.Accept()
		if err != nil {
			fmt.Println("Accept error", err)
			return
		}

		defer conn.Close()

		var signal [1]byte

		_, err = io.ReadFull(conn, signal[:])

		if err != nil {
			fmt.Println("READY receiver error", err)
			return
		}

		if signal[0] != 1 {
			fmt.Println("Invalid READY signal")
			return
		}
		// Start Receiver HostReady measurement
		hostReadyStart := time.Now()

		// B3: Buffer Allocation
		allocStart := time.Now()
		local := make([]byte, len(shared))
		allocElapsed := time.Since(allocStart)

		// B4: Shared Memory -> Local Buffer
		copyStart := time.Now()
		n := copy(local, shared)
		copyElapsed := time.Since(copyStart)

		// Receiver HostReady
		hostReadyElapsed := time.Since(hostReadyStart)

		// Validate the entire payload
		validationStart := time.Now()

		ack := byte(1)

		for i, value := range local {
			expected := byte((i + 123) % 251)

			if value != expected {
				ack = 0
				break
			}
		}

		validationElapsed := time.Since(validationStart)

		// Build ACK: status + four timing metrics
		var response [33]byte

		response[0] = ack

		binary.BigEndian.PutUint64(response[1:9],
			uint64(allocElapsed.Nanoseconds()))

		binary.BigEndian.PutUint64(response[9:17],
			uint64(copyElapsed.Nanoseconds()))

		binary.BigEndian.PutUint64(response[17:25],
			uint64(hostReadyElapsed.Nanoseconds()))

		binary.BigEndian.PutUint64(response[25:33],
			uint64(validationElapsed.Nanoseconds()))

		// Send ACK before printing results
		ackWritten, err := conn.Write(response[:])
		if err != nil {
			fmt.Println("ACK send error:", err)
			return
		}

		if ackWritten != len(response) {
			fmt.Println("Incomplete ACK write:", ackWritten)
			return
		}

		// Print results AFTER sending ACK
		fmt.Println("Receiver: READY received")

		fmt.Printf("Receiver Buffer Allocation: %.6f ms\n",
			allocElapsed.Seconds()*1000)

		fmt.Printf("Receiver Copy: %d bytes, %.6f ms\n",
			n, copyElapsed.Seconds()*1000)

		fmt.Printf("Receiver HostReady: %.6f ms\n",
			hostReadyElapsed.Seconds()*1000)

		fmt.Println("Received value:", local[0])

		fmt.Printf("Receiver Validation: %.6f ms\n",
			validationElapsed.Seconds()*1000)

		fmt.Println("Receiver: ACK sent", ack)

	case "sender":
		f, shared, err := openSharedMemory(shmPath, size)

		if err != nil {
			fmt.Println("Error: ", err)
			return
		}
		defer f.Close()
		defer syscall.Munmap(shared)

		// Prepare 64 KiB payload
		payload := make([]byte, len(shared))

		for i := range payload {
			payload[i] = byte((i + 123) % 251)
		}

		// Connect to Receiver before starting measurements
		conn, err := net.Dial("unix", "/tmp/gpuconbench-shm-test.sock")
		if err != nil {
			fmt.Println("Connection Error:", err)
			return
		}
		defer conn.Close()

		fmt.Println("Sender: Connected to Receiver")

		// End-to-End: Sender Write -> ACK received
		endToEndStart := time.Now()

		// Measure Sender Write Time
		writeStart := time.Now()

		n := copy(shared, payload)

		writeElapsed := time.Since(writeStart)

		// READY-to-ACK measurement starts
		start := time.Now()

		_, err = conn.Write([]byte{1})
		if err != nil {
			fmt.Println("READY send error:", err)
			return
		}

		var ack [33]byte

		_, err = io.ReadFull(conn, ack[:])
		if err != nil {
			fmt.Println("ACK receive error:", err)
			return
		}

		// Use the same ACK completion timestamp for both metrics
		ackReceivedAt := time.Now()

		elapsed := ackReceivedAt.Sub(start)
		endToEndElapsed := ackReceivedAt.Sub(endToEndStart)

		// Print results after measurements
		fmt.Printf("Sender: Wrote %d bytes\n", n)

		fmt.Printf("Sender Write Time: %.6f ms\n",
			writeElapsed.Seconds()*1000)

		fmt.Println("Sender: READY sent")

		if ack[0] != 1 {
			fmt.Println("Sender: Receiver validation failed")
			return
		}

		// Decode Receiver timing metrics (nanoseconds).
		allocNs := binary.BigEndian.Uint64(ack[1:9])
		copyNs := binary.BigEndian.Uint64(ack[9:17])
		hostReadyNs := binary.BigEndian.Uint64(ack[17:25])
		validationNs := binary.BigEndian.Uint64(ack[25:33])

		// Convert nanoseconds to milliseconds.
		fmt.Printf("Receiver Allocation: %.6f ms\n", float64(allocNs)/1e6)

		fmt.Printf("Receiver Copy: %.6f ms\n", float64(copyNs)/1e6)

		fmt.Printf("Receiver HostReady: %.6f ms\n", float64(hostReadyNs)/1e6)

		fmt.Printf("Receiver Validation: %.6f ms\n", float64(validationNs)/1e6)

		fmt.Printf("READY-to-ACK Latency: %.6f ms\n", elapsed.Seconds()*1000)
		fmt.Printf(
			"End-to-End Latency: %.6f ms\n",
			endToEndElapsed.Seconds()*1000,
		)

		fmt.Println("Sender: ACK received successfully")
	}
}
