package main

import (
	"encoding/binary"
	"encoding/csv"
	"flag"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"time"
)

const shmPath = "/dev/shm/gpuconbench-shm-test"

type RequestResult struct {
	WorkerID           int
	SenderWrite        time.Duration
	ReceiverAllocation time.Duration
	ReceiverCopy       time.Duration
	ReceiverHostReady  time.Duration
	ReceiverValidation time.Duration
	ReadyToACK         time.Duration
	EndToEnd           time.Duration
}

type WorkerResult struct {
	SlotIndex int
	TrialID   int
	Results   []RequestResult
	Err       error
}

type TrialSummary struct {
	TrialID       int
	TotalRequests int
	BatchElapsed  time.Duration
	Throughput    float64
}

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
	info, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, nil, err
	}

	if info.Size() != int64(size) {
		f.Close()
		return nil, nil, fmt.Errorf(
			"shared memory size mismatch: expected %d, actual %d",
			size, info.Size(),
		)
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

func parseSize(value string) (int, error) {
	var multiplier int
	var number string

	switch {
	case strings.HasSuffix(value, "KiB"):
		multiplier = 1024
		number = strings.TrimSuffix(value, "KiB")

	case strings.HasSuffix(value, "MiB"):
		multiplier = 1024 * 1024
		number = strings.TrimSuffix(value, "MiB")

	default:
		return 0, fmt.Errorf("invalid size: %s", value)
	}

	n, err := strconv.Atoi(number)
	if err != nil || n <= 0 {
		return 0, fmt.Errorf("invalid size: %s", value)
	}

	if n > int(^uint(0)>>1)/multiplier {
		return 0, fmt.Errorf("size too large: %s", value)
	}

	return n * multiplier, nil
}

func getSlot(shared []byte, slotIndex int, slotSize int) []byte {
	start := slotIndex * slotSize
	end := start + slotSize

	return shared[start:end]
}

func main() {
	fmt.Println("GPUConBench Shared Memory Benchmark")

	if len(os.Args) < 2 {
		fmt.Println("Usage: shmbench receiver|sender --size 1MiB")
		return
	}

	flags := flag.NewFlagSet(os.Args[1], flag.ContinueOnError)

	sizeArg := flags.String("size", "64KiB", "Payload size")
	warmupArg := flags.Int("warmup", 10, "Number of warmup requests")
	repeatsArg := flags.Int("repeats", 10, "Number of measurement requests")
	conArg := flags.Int("con", 1, "Number of concurrent workers")
	trialsArg := flags.Int("trials", 1, "Number of benchmark trials")

	if err := flags.Parse(os.Args[2:]); err != nil {
		return
	}
	if *conArg < 1 {
		fmt.Println("Concurrency must be >= 1")
		return
	}

	if *conArg > 256 {
		fmt.Println("Concurrency must be <= 256")
		return
	}

	fmt.Printf("Concurrency: %d\n", *conArg)

	if *warmupArg < 0 || *repeatsArg <= 0 {
		fmt.Println("Invalid counts: warmup must be >= 0 and repeats must be > 0")
		return
	}

	if *trialsArg <= 0 {
		fmt.Println("Trials must be >= 1")
		return
	}

	warmupCount := *warmupArg
	repeatCount := *repeatsArg
	trials := *trialsArg

	size, err := parseSize(*sizeArg)
	if err != nil {
		fmt.Println("Size error:", err)
		return
	}

	fmt.Printf("Payload Size: %s (%d bytes)\n", *sizeArg, size)

	// Calculate total shared memory size
	con := *conArg

	if size > int(^uint(0)>>1)/con {
		fmt.Println("Total shared memory size is too large")
		return
	}

	totalSize := size * con

	fmt.Printf("Total Shared Memory: %d bytes\n", totalSize)

	switch os.Args[1] {

	// receiver part
	case "receiver":
		f, shared, err := createSharedMemory(shmPath, totalSize)
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
		// Each connection handles its own shared memory slot.
		done := make(chan error, con)

		// Track assigned slots
		slotAssigned := make([]bool, con)
		var slotMu sync.Mutex

		requestsPerTrial := warmupCount + repeatCount

		accepted := 0
		var acceptErr error

		for i := 0; i < con; i++ {
			conn, err := listener.Accept()
			if err != nil {
				acceptErr = err
				break
			}

			accepted++

			go func(c net.Conn) {
				defer c.Close()

				// Receive Slot ID
				var slotID [1]byte

				_, err := io.ReadFull(c, slotID[:])
				if err != nil {
					done <- fmt.Errorf("slot ID receive error: %w", err)
					return
				}

				if int(slotID[0]) >= con {
					done <- fmt.Errorf("invalid slot ID: %d", slotID[0])
					return
				}

				// Check duplicate slot assignment
				index := int(slotID[0])

				slotMu.Lock()

				if slotAssigned[index] {
					slotMu.Unlock()
					done <- fmt.Errorf("duplicate slot ID: %d", index)
					return
				}

				slotAssigned[index] = true

				slotMu.Unlock()

				// Select assigned slot
				slot := getSlot(shared, index, size)

				fmt.Println("Receiver: Slot", slotID[0], "assigned")

				// Handle requests for this connection
				// Handle multiple trials on the same connection
				for trial := 0; trial < trials; trial++ {

					for j := 0; j < requestsPerTrial; j++ {

						if err := handleRequest(c, slot); err != nil {
							done <- fmt.Errorf(
								"slot %d trial %d request %d failed: %w",
								slotID[0],
								trial+1,
								j+1,
								err,
							)
							return
						}
					}
				}

				done <- nil
			}(conn)
		}

		// Wait for all accepted connections to finish
		var firstErr error

		for i := 0; i < accepted; i++ {
			if err := <-done; err != nil && firstErr == nil {
				firstErr = err
			}
		}

		if acceptErr != nil {
			fmt.Println("Accept error:", acceptErr)
		}

		if firstErr != nil {
			fmt.Println("Receiver worker error:", firstErr)
		}
	case "sender":
		f, shared, err := openSharedMemory(shmPath, totalSize)
		if err != nil {
			fmt.Println("Error:", err)
			return
		}

		defer f.Close()
		defer syscall.Munmap(shared)

		workerDone := make(chan WorkerResult, con)
		workerReady := make(chan int, con)

		// Create one measurement barrier per trial
		starts := make([]chan struct{}, trials)
		advance := make([]chan struct{}, trials-1)

		for i := range starts {
			starts[i] = make(chan struct{})
		}

		for i := range advance {
			advance[i] = make(chan struct{})
		}

		// Launch workers
		for slotIndex := 0; slotIndex < con; slotIndex++ {
			go func(index int) {
				err := runSenderWorker(
					shared,
					size,
					index,
					warmupCount,
					repeatCount,
					trials,
					workerReady,
					starts,
					advance,
					workerDone,
				)

				if err != nil {
					workerDone <- WorkerResult{
						SlotIndex: index,
						Err:       err,
					}
				}
			}(slotIndex)
		}

		// Collect summary results across all trials
		trialSummaries := make([]TrialSummary, 0, trials)

		// Execute trials
		for trial := 0; trial < trials; trial++ {

			// Wait for every worker to finish warmup
			for i := 0; i < con; i++ {
				select {
				case <-workerReady:

				case worker := <-workerDone:
					fmt.Printf(
						"Worker %d failed before trial %d: %v\n",
						worker.SlotIndex, trial+1, worker.Err,
					)
					return
				}
			}

			fmt.Printf("\nTrial %d/%d: Starting measurement\n",
				trial+1, trials)

			// Start batch timer before releasing workers
			batchStart := time.Now()
			close(starts[trial])

			// Collect results for this trial
			results := make([]RequestResult, 0, con*repeatCount)

			for i := 0; i < con; i++ {
				worker := <-workerDone

				if worker.Err != nil {
					fmt.Printf(
						"Worker %d failed: %v\n",
						worker.SlotIndex, worker.Err,
					)
					return
				}

				if worker.TrialID != trial+1 {
					fmt.Printf(
						"Unexpected trial ID: %d\n",
						worker.TrialID,
					)
					return
				}

				for _, result := range worker.Results {
					result.WorkerID = worker.SlotIndex
					results = append(results, result)
				}
			}

			batchElapsed := time.Since(batchStart)

			totalBytes := float64(size) * float64(len(results))
			throughputMiBps := (totalBytes / (1024 * 1024)) /
				batchElapsed.Seconds()

			fmt.Printf("Batch Elapsed: %.6f ms\n",
				batchElapsed.Seconds()*1000)

			fmt.Printf("Aggregate Throughput: %.2f MiB/s\n",
				throughputMiBps)

			fmt.Printf("Measurements collected: %d\n", len(results))

			// Save detailed CSV for this trial
			csvPath, err := saveResultsCSV(
				results, *sizeArg, con, trial+1,
			)
			if err != nil {
				fmt.Println("CSV save error:", err)
				return
			}

			// Save summary CSV for this trial
			summaryPath, err := saveSummaryCSV(
				csvPath,
				*sizeArg,
				con,
				warmupCount,
				repeatCount,
				batchElapsed,
				throughputMiBps,
			)
			if err != nil {
				fmt.Println("Summary CSV save error:", err)
				return
			}
			fmt.Println("CSV saved:", csvPath)
			fmt.Println("Summary CSV saved:", summaryPath)

			// Store this trial's summary
			trialSummaries = append(trialSummaries, TrialSummary{
				TrialID:       trial + 1,
				TotalRequests: len(results),
				BatchElapsed:  batchElapsed,
				Throughput:    throughputMiBps,
			})

			// Allow workers to begin the next trial
			if trial < trials-1 {
				close(advance[trial])
			}
		}
		// Save consolidated summary after all trials
		runSummaryPath, err := saveRunSummaryCSV(
			trialSummaries,
			*sizeArg,
			con,
			warmupCount,
			repeatCount,
		)

		if err != nil {
			fmt.Println("Run summary CSV save error:", err)
			return
		}

		fmt.Println("\nAll trials completed.")
		fmt.Println("Run Summary CSV saved:", runSummaryPath)
	}

}

func handleRequest(conn net.Conn, shared []byte) error {
	var signal [1]byte

	_, err := io.ReadFull(conn, signal[:])

	if err != nil {
		return fmt.Errorf("READY receive error: %w", err)
	}

	if signal[0] != 1 {
		return fmt.Errorf("invalid READY signal: %d", signal[0])
	}
	// Start Receiver HostReady measurement
	hostReadyStart := time.Now()

	// B3: Buffer Allocation
	allocStart := time.Now()
	local := make([]byte, len(shared))
	allocElapsed := time.Since(allocStart)

	// B4: Shared Memory -> Local Buffer
	copyStart := time.Now()
	copy(local, shared)
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
		return fmt.Errorf("ACK send error: %w", err)
	}

	if ackWritten != len(response) {

		return fmt.Errorf(
			"incomplete ACK write: %d/%d",
			ackWritten, len(response),
		)
	}

	return nil
}

func sendRequest(conn net.Conn, shared []byte, payload []byte) (RequestResult, error) {

	// End-to-End: Sender Write -> ACK received
	endToEndStart := time.Now()

	// Measure Sender Write Time
	writeStart := time.Now()

	copy(shared, payload)

	writeElapsed := time.Since(writeStart)

	// READY-to-ACK measurement starts
	start := time.Now()

	_, err := conn.Write([]byte{1})
	if err != nil {
		return RequestResult{}, fmt.Errorf("READY send error: %w", err)
	}

	var ack [33]byte

	_, err = io.ReadFull(conn, ack[:])
	if err != nil {
		return RequestResult{}, fmt.Errorf("ACK receive error: %w", err)
	}

	// Use the same ACK completion timestamp for both metrics
	ackReceivedAt := time.Now()

	elapsed := ackReceivedAt.Sub(start)
	endToEndElapsed := ackReceivedAt.Sub(endToEndStart)

	if ack[0] != 1 {
		return RequestResult{}, fmt.Errorf("receiver validation failed: ACK=%d", ack[0])
	}

	// Decode Receiver timing metrics (nanoseconds).
	allocNs := binary.BigEndian.Uint64(ack[1:9])
	copyNs := binary.BigEndian.Uint64(ack[9:17])
	hostReadyNs := binary.BigEndian.Uint64(ack[17:25])
	validationNs := binary.BigEndian.Uint64(ack[25:33])

	result := RequestResult{
		SenderWrite:        writeElapsed,
		ReceiverAllocation: time.Duration(allocNs),
		ReceiverCopy:       time.Duration(copyNs),
		ReceiverHostReady:  time.Duration(hostReadyNs),
		ReceiverValidation: time.Duration(validationNs),
		ReadyToACK:         elapsed,
		EndToEnd:           endToEndElapsed,
	}

	return result, nil

}

func saveResultsCSV(results []RequestResult, sizeLabel string, con int, trialID int) (string, error) {
	dir := "results/shm"

	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}

	path := fmt.Sprintf(
		"%s/shm_%s_con%d_trial%02d_%s.csv",
		dir,
		sizeLabel,
		con,
		trialID,
		time.Now().Format("20060102_150405"),
	)

	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	writer := csv.NewWriter(f)

	header := []string{
		"request_id",
		"worker_id",
		"sender_write_ms",
		"receiver_allocation_ms",
		"receiver_copy_ms",
		"receiver_hostready_ms",
		"receiver_validation_ms",
		"ready_to_ack_ms",
		"end_to_end_ms",
	}

	if err := writer.Write(header); err != nil {
		return "", err
	}

	for i, r := range results {
		record := []string{
			fmt.Sprint(i + 1),
			fmt.Sprint(r.WorkerID),
			fmt.Sprintf("%.6f", r.SenderWrite.Seconds()*1000),
			fmt.Sprintf("%.6f", r.ReceiverAllocation.Seconds()*1000),
			fmt.Sprintf("%.6f", r.ReceiverCopy.Seconds()*1000),
			fmt.Sprintf("%.6f", r.ReceiverHostReady.Seconds()*1000),
			fmt.Sprintf("%.6f", r.ReceiverValidation.Seconds()*1000),
			fmt.Sprintf("%.6f", r.ReadyToACK.Seconds()*1000),
			fmt.Sprintf("%.6f", r.EndToEnd.Seconds()*1000),
		}

		if err := writer.Write(record); err != nil {
			return "", err
		}
	}

	writer.Flush()

	if err := writer.Error(); err != nil {
		return "", err
	}

	return path, nil
}

func runSenderWorker(
	shared []byte,
	size int,
	slotIndex int,
	warmupCount int,
	repeatCount int,
	trials int,
	ready chan<- int,
	starts []chan struct{},
	advance []chan struct{},
	workerDone chan<- WorkerResult,
) error {

	if slotIndex < 0 || slotIndex > 255 {
		return fmt.Errorf("invalid slot index: %d", slotIndex)
	}

	if size <= 0 || slotIndex >= len(shared)/size {
		return fmt.Errorf("slot %d is out of range", slotIndex)
	}

	slot := getSlot(shared, slotIndex, size)

	// Prepare payload once
	payload := make([]byte, len(slot))

	for i := range payload {
		payload[i] = byte((i + 123) % 251)
	}

	// Connect once and reuse the connection across trials
	conn, err := net.Dial("unix", "/tmp/gpuconbench-shm-test.sock")
	if err != nil {
		return fmt.Errorf("connection error: %w", err)
	}
	defer conn.Close()

	// Send Slot ID
	n, err := conn.Write([]byte{byte(slotIndex)})
	if err != nil {
		return fmt.Errorf("slot ID send error: %w", err)
	}

	if n != 1 {
		return fmt.Errorf("incomplete slot ID write: %d/1", n)
	}

	// Execute multiple trials
	for trial := 0; trial < trials; trial++ {

		results, err := runOneTrial(
			conn,
			slot,
			payload,
			slotIndex,
			warmupCount,
			repeatCount,
			ready,
			starts[trial],
		)

		if err != nil {
			return fmt.Errorf("trial %d failed: %w", trial+1, err)
		}

		// Report this trial's results
		workerDone <- WorkerResult{
			SlotIndex: slotIndex,
			TrialID:   trial + 1,
			Results:   results,
		}

		// Wait until main finishes processing this trial
		if trial < trials-1 {
			<-advance[trial]
		}
	}

	return nil
}

func saveSummaryCSV(
	detailCSVPath string,
	sizeLabel string,
	con int,
	warmupCount int,
	repeatCount int,
	batchElapsed time.Duration,
	throughputMiBps float64,
) (string, error) {

	// Create summary directory
	dir := "results/shm/summary"

	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}

	// Use the same filename as the detailed CSV
	path := filepath.Join(
		dir,
		filepath.Base(detailCSVPath),
	)

	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	writer := csv.NewWriter(f)

	// CSV Header
	header := []string{
		"payload_size",
		"concurrency",
		"warmup",
		"repeats",
		"total_requests",
		"batch_elapsed_ms",
		"aggregate_throughput_mib_s",
	}

	if err := writer.Write(header); err != nil {
		return "", err
	}

	// One row per benchmark run
	record := []string{
		sizeLabel,
		strconv.Itoa(con),
		strconv.Itoa(warmupCount),
		strconv.Itoa(repeatCount),
		strconv.Itoa(con * repeatCount),
		fmt.Sprintf("%.6f", batchElapsed.Seconds()*1000),
		fmt.Sprintf("%.6f", throughputMiBps),
	}

	if err := writer.Write(record); err != nil {
		return "", err
	}

	writer.Flush()

	if err := writer.Error(); err != nil {
		return "", err
	}

	return path, nil
}
func runOneTrial(
	conn net.Conn,
	slot []byte,
	payload []byte,
	slotIndex int,
	warmupCount int,
	repeatCount int,
	ready chan<- int,
	start <-chan struct{},
) ([]RequestResult, error) {

	// Warmup
	for i := 0; i < warmupCount; i++ {
		_, err := sendRequest(conn, slot, payload)
		if err != nil {
			return nil, fmt.Errorf(
				"warmup %d failed: %w", i+1, err,
			)
		}
	}

	// Notify main that warmup is complete
	ready <- slotIndex

	// Wait for measurement start signal
	<-start

	// Measurement
	results := make([]RequestResult, 0, repeatCount)

	for i := 0; i < repeatCount; i++ {
		result, err := sendRequest(conn, slot, payload)
		if err != nil {
			return nil, fmt.Errorf(
				"measurement %d failed: %w", i+1, err,
			)
		}

		results = append(results, result)
	}

	return results, nil
}

func saveRunSummaryCSV(
	summaries []TrialSummary,
	sizeLabel string,
	con int,
	warmupCount int,
	repeatCount int,
) (string, error) {

	dir := "results/shm/summary"

	if err := os.MkdirAll(dir, 0755); err != nil {
		return "", err
	}

	path := fmt.Sprintf(
		"%s/shm_%s_con%d_run_summary_%s.csv",
		dir,
		sizeLabel,
		con,
		time.Now().Format("20060102_150405.000000000"),
	)

	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()

	writer := csv.NewWriter(f)

	header := []string{
		"trial_id",
		"payload_size",
		"concurrency",
		"warmup",
		"repeats",
		"total_requests",
		"batch_elapsed_ms",
		"aggregate_throughput_mib_s",
	}

	if err := writer.Write(header); err != nil {
		return "", err
	}

	for _, s := range summaries {
		record := []string{
			strconv.Itoa(s.TrialID),
			sizeLabel,
			strconv.Itoa(con),
			strconv.Itoa(warmupCount),
			strconv.Itoa(repeatCount),
			strconv.Itoa(s.TotalRequests),
			fmt.Sprintf("%.6f", s.BatchElapsed.Seconds()*1000),
			fmt.Sprintf("%.6f", s.Throughput),
		}

		if err := writer.Write(record); err != nil {
			return "", err
		}
	}

	writer.Flush()

	if err := writer.Error(); err != nil {
		return "", err
	}

	return path, nil
}
