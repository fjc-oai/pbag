def my_raise():
    raise Exception('my_raise')

def fn():
    try:
        my_raise()
    finally:
        print('finally')

def main():
    fn()

if __name__ == '__main__':
    main()

