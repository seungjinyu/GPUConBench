package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/spf13/cobra"
)

const (
	address         = "127.0.0.1:5002"
	headerSize      = 16
	maxPayloadBytes = 1024 * 1024 * 16
)

func main() {

	root := &cobra.Command{
		Use: "gpuconbench",
	}

	receiver := &cobra.Command{
		Use:   "receiver",
		Short: " Start tcp receiver",
		RunE: func(cmd *cobra.Command, args []string) error {
			return runReceiver()
		},
	}

	sender := &cobra.Command{

		Use: "sender",
		RunE: func(cmd *cobra.Command, args []string) error {
			con, _ := cmd.Flags().GetInt("con")
			payloadKiB, _ := cmd.Flags().GetInt("payload-kib")

			return runSender(con, payloadKiB)
		},
	}

	sender.Flags().Int("con", 1, "Concurrency")
	sender.Flags().Int("payload-kib", 64, "Payload size in KiB")

	root.AddCommand(sender)
	root.AddCommand(receiver)

	cobra.CheckErr(root.Execute())
}

func runReceiver() error {

	fmt.Println("Receiver started")
	listener, err := net.Listen("tcp", address)
	if err != nil {
		return err
	}
	defer listener.Close()

	fmt.Println("Receiver listening on ", address)

	for {
		conn, err := listener.Accept()
		if err != nil {
			return err
		}

		acceptedAt := time.Now()
		go handleConnection(conn, acceptedAt)
	}
}

func handleConnection(conn net.Conn, acceptedAt time.Time) {

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

	requestID := binary.BigEndian.Uint64(header[0:8])
	payloadLen := binary.BigEndian.Uint64(header[8:16])

	if payloadLen == 0 || payloadLen > maxPayloadBytes {
		fmt.Printf("Invalid payload length: %d\n", payloadLen)
		return
	}

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

	for i, b := range data {
		if b != byte(i%251) {
			fmt.Printf("Request %d: data mismatch at %d\n", requestID, i)
			return
		}
	}

	fmt.Printf(
		"Received request=%d payload=%d bytes\n",
		requestID,
		len(data),
	)
	fmt.Printf(
		"Request=%d Header=%.3fms Alloc=%.3fms Recv=%.3fms HostReady=%.3fms\n",
		requestID,
		headerEnd.Sub(headerStart).Seconds()*1000,
		allocEnd.Sub(allocStart).Seconds()*1000,
		recvEnd.Sub(recvStart).Seconds()*1000,
		recvEnd.Sub(acceptedAt).Seconds()*1000,
	)

	var ack [11]byte
	copy(ack[0:3], "ACK")
	binary.BigEndian.PutUint64(ack[3:11], requestID)

	if err := writeAll(conn, ack[:]); err != nil {
		fmt.Println("ACK send error:", err)
	}

}

func runSender(con int, payloadKiB int) error {

	if con < 1 || con > 128 {
		return fmt.Errorf("con must be between 1 and 128")
	}

	if payloadKiB < 1 || payloadKiB > 16*1024 {
		return fmt.Errorf("payload-kib must be between 1 and 16384")
	}

	payload := make([]byte, payloadKiB*1024)
	// 왜 251?
	for i := range payload {
		payload[i] = byte(i % 251)
	}

	var wg sync.WaitGroup

	start := make(chan struct{})
	results := make(chan error, con)

	for i := 0; i < con; i++ {
		wg.Add(1)

		go func(id int) {
			defer wg.Done()
			<-start

			err := sendRequest(id, payload)
			if err != nil {
				results <- fmt.Errorf("Request %d: %w", id, err)
				return
			}
			results <- nil
		}(i)
	}
	fmt.Printf(
		"Starting CON %d payload=%d bytes\n",
		con,
		len(payload),
	)

	close(start)

	wg.Wait()
	close(results)

	success := 0
	failed := 0

	for err := range results {
		if err != nil {
			fmt.Println("Error:", err)
			failed++
		} else {
			success++
		}
	}
	fmt.Printf(
		"Completed: %d success, %d failed\n",
		success,
		failed,
	)
	if failed > 0 {
		return fmt.Errorf("%d requests failed", failed)
	}

	return nil
}

func sendRequest(id int, payload []byte) error {
	conn, err := net.DialTimeout(
		"tcp",
		address,
		5*time.Second,
	)
	if err != nil {
		return err
	}
	defer conn.Close()

	if err := conn.SetDeadline(time.Now().Add(10 * time.Second)); err != nil {
		return err
	}

	var header [headerSize]byte
	binary.BigEndian.PutUint64(header[0:8], uint64(id))
	binary.BigEndian.PutUint64(header[8:16], uint64(len(payload)))

	if err := writeAll(conn, header[:]); err != nil {
		return err
	}

	if err := writeAll(conn, payload); err != nil {
		return err
	}

	var ack [11]byte
	_, err = io.ReadFull(conn, ack[:])
	if err != nil {
		return err
	}
	if string(ack[0:3]) != "ACK" {
		return fmt.Errorf("invalid ACK")
	}

	ackID := binary.BigEndian.Uint64(ack[3:11])

	if ackID != uint64(id) {
		return fmt.Errorf("ACK ID mismatch: expected %d, got %d", id, ackID)
	}

	return nil
}

func writeAll(conn net.Conn, data []byte) error {

	for len(data) > 0 {

		n, err := conn.Write(data)
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
