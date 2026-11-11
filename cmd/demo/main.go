package main

import (
	"encoding/binary"
	"encoding/csv"
	"fmt"
	"io"
	"net"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"sync"
	"time"

	"github.com/spf13/cobra"
)

const (
	address         = "192.168.250.1:5002"
	headerSize      = 16
	ackSize         = 68
	maxPayloadBytes = 1024 * 1024 * 16
)

type StageMetrics struct {
	AcceptToHandler time.Duration
	HeaderRead      time.Duration
	HeaderParse     time.Duration
	BufferAlloc     time.Duration
	PayloadRead     time.Duration
	HostReady       time.Duration
	Validation      time.Duration
}

type RequestResult struct {
	ID           int
	RoundID      int
	BatchElapsed time.Duration

	Dial     time.Duration
	RTT      time.Duration
	Verified bool
	Stages   StageMetrics
	Err      error
}

func main() {

	root := &cobra.Command{
		Use: "gpuconbench",
	}

	receiver := &cobra.Command{
		Use:   "receiver",
		Short: " Start tcp receiver",
		RunE: func(cmd *cobra.Command, args []string) error {
			verify, err := cmd.Flags().GetBool("verify")
			if err != nil {
				return err
			}
			return runReceiver(verify)
		},
	}
	receiver.Flags().Bool("verify", true, "Validate test payload bytes")

	sender := &cobra.Command{

		Use: "sender",
		RunE: func(cmd *cobra.Command, args []string) error {
			con, err := cmd.Flags().GetInt("con")
			if err != nil {
				return err
			}
			payloadKiB, err := cmd.Flags().GetInt("payload-kib")
			if err != nil {
				return err
			}
			output, err := cmd.Flags().GetString("output")
			if err != nil {
				return err
			}
			warmup, err := cmd.Flags().GetInt("warmup")
			if err != nil {
				return err
			}
			repeats, err := cmd.Flags().GetInt("repeats")
			if err != nil {
				return err
			}

			return runSender(con, payloadKiB, warmup, repeats, output)
		},
	}

	sender.Flags().Int("con", 1, "Concurrency")
	sender.Flags().Int("payload-kib", 64, "Payload size in KiB")
	sender.Flags().String("output", "", "CSV path (default: ")

	sender.Flags().Int("warmup", 20, "Number of warmup rounds")
	sender.Flags().Int("repeats", 100, "Number of measurement rounds")

	root.AddCommand(sender)
	root.AddCommand(receiver)

	cobra.CheckErr(root.Execute())
}

func runReceiver(verify bool) error {

	fmt.Println("Receiver started")
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	defer listener.Close()

	fmt.Printf(
		"Receiver listening on %s (verify=%v)\n",
		address,
		verify,
	)

	for {
		conn, err := listener.Accept()
		if err != nil {
			return err
		}

		acceptedAt := time.Now()
		go handleConnection(conn, acceptedAt, verify)
	}
}

func handleConnection(conn net.Conn, acceptedAt time.Time, verify bool) {

	handlerStart := time.Now()
	defer conn.Close()

	if err := conn.SetDeadline(time.Now().Add(30 * time.Second)); err != nil {
		fmt.Println("Deadline error:", err)
		return
	}

	var header [headerSize]byte

	headerStart := time.Now()

	_, err := io.ReadFull(conn, header[:])
	if err != nil {
		fmt.Println("Header receive error:", err)
		return
	}
	headerEnd := time.Now()

	parseStart := time.Now()

	requestID := binary.BigEndian.Uint64(header[0:8])
	payloadLen := binary.BigEndian.Uint64(header[8:16])

	if payloadLen == 0 || payloadLen > maxPayloadBytes {
		fmt.Printf("Invalid payload length: %d\n", payloadLen)
		return
	}
	parseEnd := time.Now()

	allocStart := time.Now()
	data := make([]byte, int(payloadLen))
	allocEnd := time.Now()

	recvStart := time.Now()
	_, err = io.ReadFull(conn, data)
	if err != nil {
		fmt.Println("Payload receive error:", err)
		return
	}
	recvEnd := time.Now()

	verifyStart := time.Now()

	if verify {
		for i, b := range data {
			if b != byte(i%251) {
				fmt.Printf("Request %d: data mismatch at %d\n", requestID, i)
				return
			}
		}
	}

	verifyEnd := time.Now()

	metrics := StageMetrics{
		AcceptToHandler: handlerStart.Sub(acceptedAt),
		HeaderRead:      headerEnd.Sub(headerStart),
		HeaderParse:     parseEnd.Sub(parseStart),
		BufferAlloc:     allocEnd.Sub(allocStart),
		PayloadRead:     recvEnd.Sub(recvStart),
		HostReady:       recvEnd.Sub(acceptedAt),
		Validation:      verifyEnd.Sub(verifyStart),
	}

	if err := sendACK(conn, requestID, verify, metrics); err != nil {
		fmt.Printf("Request %d: ACK send error: %v\n", requestID, err)
	}

}

func sendACK(conn net.Conn, id uint64, verified bool, m StageMetrics) error {
	var ack [ackSize]byte
	copy(ack[:3], "ACK")

	binary.BigEndian.PutUint64(ack[3:11], id)
	if verified {
		ack[11] = 1
	}
	values := [...]time.Duration{
		m.AcceptToHandler,
		m.HeaderRead,
		m.HeaderParse,
		m.BufferAlloc,
		m.PayloadRead,
		m.HostReady,
		m.Validation,
	}
	for i, v := range values {
		binary.BigEndian.PutUint64(
			ack[12+i*8:20+i*8],
			uint64(v.Nanoseconds()),
		)
	}

	return writeAll(conn, ack[:])
}

func runSender(
	con int,
	payloadKiB int,
	warmup int,
	repeats int,
	output string,
) error {

	if con < 1 || con > 128 {
		return fmt.Errorf("con must be between 1 and 128")
	}

	if payloadKiB < 1 || payloadKiB > 16*1024 {
		return fmt.Errorf("payload-kib must be between 1 and 16384")
	}
	if warmup < 0 {
		return fmt.Errorf("warmup must be >= 0")
	}

	if repeats < 1 {
		return fmt.Errorf("repeats must be >= 1")
	}

	payload := make([]byte, payloadKiB*1024)
	// 왜 251?
	for i := range payload {
		payload[i] = byte(i % 251)
	}
	fmt.Printf(
		"Starting CON %d payload=%d bytes\n",
		con,
		len(payload),
	)
	// Warmup 실행
	for i := 0; i < warmup; i++ {
		warmupResults, _ := runRound(con, payload, -1)

		for _, r := range warmupResults {
			if r.Err != nil {
				return fmt.Errorf(
					"warmup %d, request %d failed: %w",
					i, r.ID, r.Err,
				)
			}
		}
	}

	// 실제 측정 결과를 모을 Slice
	results := make([]RequestResult, 0, con*repeats)

	// 각 Round의 실행 시간 합계
	var batchElapsed time.Duration

	for roundID := 0; roundID < repeats; roundID++ {

		roundResults, elapsed := runRound(con, payload, roundID)

		results = append(results, roundResults...)

		batchElapsed += elapsed
	}

	// 성공한 요청 계산
	successes := 0

	for _, r := range results {
		if r.Err == nil {
			successes++
		} else {
			fmt.Printf("Request %d: %v\n", r.ID, r.Err)
		}
	}

	sort.Slice(results, func(i, j int) bool {
		if results[i].RoundID != results[j].RoundID {
			return results[i].RoundID < results[j].RoundID
		}

		return results[i].ID < results[j].ID
	})

	if output == "" {
		output = fmt.Sprintf(
			"results/raw/con%d_%dKiB_%s.csv",
			con,
			payloadKiB,
			time.Now().Format("20060102_150405.000000000"),
		)
	}

	if err := writeCSV(output, con, len(payload), results); err != nil {
		return err
	}
	throughputMiBs := float64(successes*len(payload)) /
		(1024 * 1024) / batchElapsed.Seconds()

	totalRequests := con * repeats

	fmt.Printf("Completed: %d success, %d failed\n",
		successes, totalRequests-successes)
	fmt.Printf("Total Batch=%.3f ms\n", ms(batchElapsed))

	fmt.Printf(
		"Average Batch=%.3f ms\n",
		ms(batchElapsed)/float64(repeats),
	)

	fmt.Printf(
		"Effective throughput=%.2f MiB/s\n",
		throughputMiBs,
	)
	// 성공한 요청의 HostReady Latency 수집
	var hostReady []float64

	for _, r := range results {
		if r.Err == nil {
			hostReady = append(hostReady, ms(r.Stages.HostReady))
		}
	}

	// Latency를 오름차순으로 정렬
	sort.Float64s(hostReady)

	if len(hostReady) > 0 {

		// Nearest-rank percentile
		percentile := func(p int) float64 {
			index := (p*len(hostReady)+99)/100 - 1
			return hostReady[index]
		}

		fmt.Printf(
			"HostReady P50=%.3f ms P95=%.3f ms P99=%.3f ms\n",
			percentile(50),
			percentile(95),
			percentile(99),
		)
	}

	fmt.Println("CSV:", output)

	if successes != totalRequests {
		return fmt.Errorf("%d requests failed", totalRequests-successes)
	}
	return nil
}

func runRound(
	con int,
	payload []byte,
	roundID int,
) ([]RequestResult, time.Duration) {

	var wg sync.WaitGroup

	start := make(chan struct{})
	resultCh := make(chan RequestResult, con)

	for i := 0; i < con; i++ {
		wg.Add(1)

		go func(id int) {
			defer wg.Done()

			<-start

			result := sendRequest(id, payload)
			result.RoundID = roundID

			resultCh <- result
		}(i)
	}

	batchStart := time.Now()
	close(start)

	wg.Wait()
	batchElapsed := time.Since(batchStart)

	close(resultCh)

	results := make([]RequestResult, 0, con)

	for result := range resultCh {
		result.BatchElapsed = batchElapsed
		results = append(results, result)
	}

	return results, batchElapsed
}

func sendRequest(id int, payload []byte) RequestResult {
	result := RequestResult{ID: id}

	dialStart := time.Now()
	conn, err := net.DialTimeout("tcp", address, 5*time.Second)
	result.Dial = time.Since(dialStart)

	if err != nil {
		result.Err = err
		return result
	}
	defer conn.Close()

	if err := conn.SetDeadline(time.Now().Add(30 * time.Second)); err != nil {
		result.Err = err
		return result
	}

	var header [headerSize]byte
	binary.BigEndian.PutUint64(header[0:8], uint64(id))
	binary.BigEndian.PutUint64(header[8:16], uint64(len(payload)))

	requestStart := time.Now()

	if err := writeAll(conn, header[:]); err != nil {
		result.Err = err
		return result
	}
	if err := writeAll(conn, payload); err != nil {
		result.Err = err
		return result
	}

	// ACK + Receiver timing 수신
	var ack [ackSize]byte
	if _, err := io.ReadFull(conn, ack[:]); err != nil {
		result.Err = err
		return result
	}

	result.RTT = time.Since(requestStart)

	if string(ack[:3]) != "ACK" {
		result.Err = fmt.Errorf("invalid ACK")
		return result
	}

	gotID := binary.BigEndian.Uint64(ack[3:11])
	if gotID != uint64(id) {
		result.Err = fmt.Errorf("ACK ID mismatch: got %d", gotID)
		return result
	}

	result.Verified = ack[11] == 1

	d := func(i int) time.Duration {
		return time.Duration(binary.BigEndian.Uint64(
			ack[12+i*8 : 20+i*8],
		))
	}

	result.Stages = StageMetrics{
		AcceptToHandler: d(0),
		HeaderRead:      d(1),
		HeaderParse:     d(2),
		BufferAlloc:     d(3),
		PayloadRead:     d(4),
		HostReady:       d(5),
		Validation:      d(6),
	}

	return result
}

// -----------------------------
// CSV 저장
// -----------------------------

func writeCSV(
	path string,
	con, payloadBytes int,
	results []RequestResult,
) error {
	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return err
	}

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	w := csv.NewWriter(f)

	if err := w.Write([]string{
		"request_id", "con", "payload_bytes",
		"dial_ms", "request_rtt_ms",
		"accept_to_handler_ms", "header_recv_ms",
		"header_parse_ms", "buffer_alloc_ms",
		"payload_recv_ms", "host_ready_ms",
		"validation_ms", "verified",
		"batch_elapsed_ms", "status", "error", "round_id",
	}); err != nil {
		return err
	}

	fms := func(d time.Duration) string {
		return strconv.FormatFloat(ms(d), 'f', 6, 64)
	}

	for _, r := range results {
		// 실패 요청은 측정되지 않은 값을 0이 아닌 빈칸으로 저장
		row := []string{
			strconv.Itoa(r.ID),
			strconv.Itoa(con),
			strconv.Itoa(payloadBytes),
			fms(r.Dial), "", "", "", "", "", "", "", "", "",
			fms(r.BatchElapsed), "error", "", strconv.Itoa(r.RoundID),
		}

		if r.Err != nil {
			row[15] = r.Err.Error()
		} else {
			row[4] = fms(r.RTT)
			row[5] = fms(r.Stages.AcceptToHandler)
			row[6] = fms(r.Stages.HeaderRead)
			row[7] = fms(r.Stages.HeaderParse)
			row[8] = fms(r.Stages.BufferAlloc)
			row[9] = fms(r.Stages.PayloadRead)
			row[10] = fms(r.Stages.HostReady)
			row[11] = fms(r.Stages.Validation)
			row[12] = strconv.FormatBool(r.Verified)
			row[14] = "ok"
		}

		if err := w.Write(row); err != nil {
			return err
		}
	}

	w.Flush()
	return w.Error()
}

func ms(d time.Duration) float64 {
	return float64(d) / float64(time.Millisecond)
}

func writeAll(w io.Writer, data []byte) error {
	for len(data) > 0 {
		n, err := w.Write(data)
		if err != nil {
			return err
		}
		if n == 0 {
			return io.ErrShortWrite
		}
		data = data[n:]
	}
	return nil
}
