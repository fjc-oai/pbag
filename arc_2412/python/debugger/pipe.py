import multiprocessing

def producer(pipe):
    print("producer")
    pipe.send('halo')
    pipe.close()

def main():
    parent_conn, child_conn = multiprocessing.Pipe()

    p = multiprocessing.Process(target=producer, args=(child_conn,))
    p.start()
    p.join()

    res = parent_conn.recv()
    print(res)
    parent_conn.close()

if __name__ == "__main__":
    main()
