import sys


def add_space_and_dot_to_lines(input_file: str, output_file: str) -> None:
    """
    Reads a text file, adds a space and a dot at the end of each non-empty line,
    and writes the result to another file.
    """
    with (
        open(input_file, "r", encoding="utf-8") as infile,
        open(output_file, "w", encoding="utf-8") as outfile,
    ):
        for line in infile:
            line = line.strip()
            if line:  # keep empty lines as empty
                outfile.write(line + " .\n")
            else:
                outfile.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python post_process.py <input_file> <output_file>")
        sys.exit(1)

    add_space_and_dot_to_lines(sys.argv[1], sys.argv[2])
