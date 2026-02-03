import socket
import threading
import asyncio


def handle_client(conn, addr):
    print("Client connected:", addr)
    while True:
        data = conn.recv(1024)
        if not data:
            break

        msg = data.decode()
        print(f"[{addr}] {msg}")

        if msg == "exit":
            conn.sendall(b"exiting......")
            break
        else:
            conn.sendall(b"ACK")

    conn.close()
    print("Client disconnected:", addr)


def Server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("0.0.0.0", 9000))
    server.listen(5)

    print("Server listening on 9000")

    while True:
        conn, addr = server.accept()
        t = threading.Thread(
            target=handle_client,
            args=(conn, addr),
            daemon=True
        )
        t.start()


async def handle_client2(reader, writer):
    addr = writer.get_extra_info("peername")
    print("Client:", addr)

    while True:
        data = await reader.read(1024)
        if not data:
            break

        msg = data.decode()
        print(f"[{addr}] {msg}")

        if msg == "exit":
            writer.write(b"exiting......")
            await writer.drain()
            break
        else:
            writer.write(b"ACK")
            await writer.drain()

    writer.close()
    await writer.wait_closed()


async def main():
    server = await asyncio.start_server(
        handle_client2, "0.0.0.0", 9000
    )
    async with server:
        await server.serve_forever()

# asyncio.run(main())


if __name__ == "__main__":
    Server()
