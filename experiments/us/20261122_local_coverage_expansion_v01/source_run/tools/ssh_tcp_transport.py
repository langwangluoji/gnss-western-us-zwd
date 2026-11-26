"""Per-command SSH byte transport; no credential handling or persistent settings."""
import socket,sys,threading


def main():
    stream=socket.create_connection((sys.argv[1],int(sys.argv[2])),timeout=20)
    stream.settimeout(None)
    def write():
        try:
            while True:
                chunk=sys.stdin.buffer.read1(65536)
                if not chunk: break
                stream.sendall(chunk)
            stream.shutdown(socket.SHUT_WR)
        except OSError: pass
    threading.Thread(target=write,daemon=True).start()
    try:
        while True:
            chunk=stream.recv(65536)
            if not chunk: break
            sys.stdout.buffer.write(chunk); sys.stdout.buffer.flush()
    finally: stream.close()


if __name__=='__main__': main()
