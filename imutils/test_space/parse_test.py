import sys

print(sys.argv[1])

def my_func1(arg1, arg2):
    return arg1*arg2

def my_func2(arg1, arg2, arg3):
    return arg1*arg2*arg3

def my_func3(arg1):
    return print(arg1)

print(sys.argv[2:5])

fun="%s(*%s)" % (sys.argv[1],sys.argv[2:])

print(fun)
eval(fun)