package mai

import (
	"fmt"
	"net"
)



func main(){
	fmt.Println("Hello world.")

	conn , err := net.Dial("tcp","127.0.0.1:5001")

	if err != nil {
		panic(err)
	}

	defer conn.Close()

	message := []byte("Hello GPU")

	_,err = conn.Write(message)

	if err != nil {
		panic(err)
	}
	
}