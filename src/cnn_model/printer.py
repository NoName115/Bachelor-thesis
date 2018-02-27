
def print_info(message, tabs=0):
    print(
        (' ' * tabs * 2) +
        "[INFO] " + message
    )

def print_warning(message, tabs=0):
    print(
        (' ' * tabs * 2) +
        "[! WARNING !] " + message
    )

def print_error(message):
    print("[! ERROR !] " + message)
    exit(-1)
