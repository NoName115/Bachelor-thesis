
def print_blank(message, spaces=0):
    print(
        (' ' * spaces * 2) +
        str(message)
    )

def print_info(message, spaces=0):
    print(
        (' ' * spaces * 2) +
        "[INFO] " + str(message)
    )

def print_warning(message, spaces=0):
    print(
        (' ' * spaces * 2) +
        "[! WARNING !] " + str(message)
    )

def print_error(message):
    print("\n[! ERROR !] " + str(message) + '\n')
    exit(-1)
