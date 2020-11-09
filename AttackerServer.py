import socket

if __name__ == '__main__':
    # Create a TCP/IP socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Connect the socket to the port where the server is listening
    server_address = ('192.168.0.147', 8888)
    print('starting up on %s port %s' % server_address)
    sock.bind(server_address)
    sock.listen(1)
    f = open("attack_acc_reading.txt", "w")
    f.write("")
    f.close()

    while True:
        print('waiting for a connection...')
        connection, client_address = sock.accept()
        try:
            print('connection from', client_address)
            f = open("attack_acc_reading.txt", "a")
            while True:
                data = connection.recv(1024)
                if data:
                    f.write(data.decode('utf-8'))
                else:
                    f.close()
                    break

        finally:
            print('closing socket')
            connection.close()
