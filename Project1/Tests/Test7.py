import multiprocessing


class MyFancyClass(object):

    def __init__(self, name):
        self.name = name

    def do_something(self):
        proc_name = multiprocessing.current_process().name
        print ('Doing something fancy in %s for %s!' % (proc_name, self.name))


def worker(q1,q2):
    obj = q1.get()
    obj.do_something()

    retDic = q2.get()
    print('kkkk')


if __name__ == '__main__':
    queue1 = multiprocessing.Queue()
    queue2 = multiprocessing.Queue()

    p = multiprocessing.Process(target=worker, args=(queue1,queue2,))
    p.start()

    retDic = { 'x':5,
               'y':7}

    queue1.put(MyFancyClass('Fancy Dan'))
    queue2.put(retDic)

    # Wait for the worker to finish
    queue1.close()
    queue2.close()
    queue1.join_thread()
    queue2.join_thread()
    p.join()