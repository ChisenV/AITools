
def Client():
    import socket

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect(("127.0.0.1", 9000))

    while True:
        data = input(">> ")
        # client.sendall(data.encode())
        if data == "exit":
            # resp = client.recv(1024)
            # print(resp.decode())
            break
        else:
            client.sendall(data.encode())
            resp = client.recv(1024)
            print(resp.decode())

    client.close()


if __name__ == "__main__":
    Client()
