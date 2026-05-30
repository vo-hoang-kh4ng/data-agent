# Mock resource module for Windows compatibility

RLIMIT_CPU = 0
RLIMIT_FSIZE = 1
RLIMIT_DATA = 2
RLIMIT_STACK = 3
RLIMIT_CORE = 4
RLIMIT_RSS = 5
RLIMIT_NPROC = 6
RLIMIT_NOFILE = 7
RLIMIT_MEMLOCK = 8
RLIMIT_AS = 9

def getrlimit(resource):
    return (1024, 1024)

def setrlimit(resource, limits):
    pass

def prlimit(pid, resource, limits=None):
    return (1024, 1024)
