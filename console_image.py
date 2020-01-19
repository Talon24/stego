"""draw an image to console"""

# import numpy

RESET = '\033[0m'
def get_color_escape(r, g, b, background=False):
    return '\033[{};2;{};{};{}m'.format(48 if background else 38, r, g, b)

def draw(array, unusable, highlighted=None):
    out = []
    for row_idx, row in enumerate(array[:45]):
        for col_idx, col in enumerate(row[:180]):
            if (col_idx, row_idx) in unusable:
                begin = get_color_escape(255, 255, 0, background=True)
            else:
                begin = get_color_escape(*col[:3], background=True)
            out.append(begin)
            if (col_idx, row_idx) == highlighted:
                out.append("o")
            else:
                out.append(" ")
        out.append(RESET)
        out.append("\n")
    out.append(RESET)
    print("".join(out), end="")
