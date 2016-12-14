
a = []

def f():
    a = [1, 2]

if __name__ == '__main__':
    f()
    print(a)
    b = {'a':1, 'b':2}
    print(b.get('c', 22))