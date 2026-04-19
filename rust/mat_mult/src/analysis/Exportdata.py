from pathlib import Path


def find_mut_files(file_path_str: str):
    print("File found")
    file_path = Path(file_path_str)

    if not file_path.is_file():
        print(f"{file_path} is not a valid file.")
        return

    check_file_for_999999(file_path)


def check_file_for_999999(file_path: Path):

    try:
        # results.txt is created in the rust/mat_mult folder (one level
        # above this script's folder).
        base_dir = Path(__file__).resolve().parents[1]
        results_path = base_dir / "results.txt"

        with file_path.open("r") as f, results_path.open("w") as out:
            prev_gen_line = None
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Save "Gen ..." lines so we can print them together with the
                # corresponding "Generation ..." line.
                if line.startswith("Gen "):
                    prev_gen_line = line
                    continue

                # Lines of interest look like: "Generation 1000, fitness difference: ..., ..."
                if not line.startswith("Generation "):
                    continue

                # Extract the generation number before the first comma
                prefix = "Generation "
                rest = line[len(prefix) :]
                gen_token = rest.split(",", 1)[0].strip()

                try:
                    generation = int(gen_token)
                except ValueError:
                    continue

                # Look for generation == 4_999_999
                if generation == 4999999:
                    # Write the current "Generation ..." line and then keep
                    # writing subsequent lines until we hit the line that
                    # contains "SA power: <number>".
                    line_to_write = line
                    print(f"{file_path}: {line_to_write}")
                    out.write(line_to_write + "\n")

                    for next_line in f:
                        next_line = next_line.rstrip("\n")
                        if not next_line.strip():
                            continue
                        print(f"{file_path}: {next_line}")
                        out.write(next_line + "\n")
                        # Stop once we reach the summary line with "SA power:"
                        if "SA power:" in next_line:
                            break
    except OSError as e:
        print(f"Could not read {file_path}: {e}")


if __name__ == "__main__":
    # Change "." to the folder you want to scan, e.g. "/Users/dios/Downloads/gen_prog_mm-main/rust/mat_mult"
    find_mut_files("/Users/dios/Desktop/gen_prog_mm-main/rust/mat_mult/src/output")