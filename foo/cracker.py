from rarfile import RarFile
import threading
import queue
import time


class Worker(threading.Thread):
    def __init__(self, id, rfile, found, tasks):
        super().__init__()
        print(f'worker {id} init')
        self.id = id
        self.rfile = rfile
        self.found = found
        self.tasks = tasks

    def run(self):
        print(f'Worker {self.id} starts....')
        while not self.found:
             pwd = self.tasks.get()
             if not pwd:
                print(f'Worker done: {self.id}')
                self.tasks.task_done()
                break
             try:
                self.rfile.extractall(pwd=pwd)
                print(f'Found answer: {pwd}')
                self.found.append(pwd)
                self.tasks.task_done()
                break
             except:
                print(f'Trying next. {pwd} wrong')
                self.tasks.task_done()

class Reporter(threading.Thread):
    def __init__(self, tasks, found):
        super().__init__()
        self.tasks = tasks
        self.found = found
        self.tot = tasks.qsize()
        self.prev = self.tot
        self.prev_t = time.time()

    def run(self):
        print('reporter running')
        while not self.found:
            cur = self.tasks.qsize()
            n_consumed = self.prev - cur
            cur_t = time.time()
            dur = cur_t - self.prev_t
            print(f'QPS: {n_consumed/dur}, {cur}/{self.tot} remained')
            self.prev = cur
            self.prev_t = cur_t
            time.sleep(1)

def crack():
    print('cracking...')
    rfile = RarFile('open.rar', 'r')
    n_workers = 50
    found = []
    tasks = queue.Queue()
    for i in range(100,2000):
        tasks.put(str(i))
    for i in range(n_workers):
        tasks.put(None)

    reporter = Reporter(tasks, found)
    reporter.start()

    workers = [Worker(i, rfile, found, tasks) for i in range(n_workers)]
    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()
    reporter.join()

    print(f'Answer: {found[0]}')


def main():
    crack()


if __name__ == "__main__":
    main()
