
def print_info(message, spaces=0):
    print(
        (' ' * spaces * 2) +
        "[INFO] " + message
    )

def print_warning(message, spaces=0):
    print(
        (' ' * spaces * 2) +
        "[! WARNING !] " + message
    )

def print_error(message):
    print("[! ERROR !] " + message)
    exit(-1)
