from threading import Event, Thread


def worker(idx, wait_evt, set_evt):
    print(f'Worker {idx} started')
    for i in range(5):
        wait_evt.wait()
        print(f'Worker {idx}: {i}')
        wait_evt.clear()
        set_evt.set()
    print(f'Worker {idx} done')


def main():
    evt1 = Event()
    evt2 = Event()

    t1 = Thread(target=worker, args=(1, evt1, evt2))
    t2 = Thread(target=worker, args=(2, evt2, evt1))

    t1.start()
    t2.start()

    evt1.set()

    t1.join()
    t2.join()

if __name__ == '__main__':
    main()

