from pathlib import Path

def find_mut_files(base_dir: str):
    base_path = Path(base_dir)

    # Walk through all files under base_dir
    for file_path in base_path.rglob("*"):
        if not file_path.is_file():
            continue

        name = file_path.name
        # Select files ending with "mut10" or "mut20"
        if name.endswith("-10") or name.endswith("mut20"):
            check_file_for_999999(file_path)


def check_file_for_999999(file_path: Path):
    try:
        with file_path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # Each line: generation, average cell difference, number of different cells
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 3:
                    continue

                generation_str, avg_cell_diff_str, num_diff_cells_str = parts[:3]

                # Look for generation == 999999
                try:
                    generation = int(generation_str)
                except ValueError:
                    continue

                if generation == 999999:
                    print(
                        f"{file_path}: "
                        f"avg_cell_diff={avg_cell_diff_str}, "
                        f"num_diff_cells={num_diff_cells_str}"
                    )
    except OSError as e:
        print(f"Could not read {file_path}: {e}")


if __name__ == "__main__":
    # Use folder containing this script so it works regardless of cwd
    base_dir = Path(__file__).parent / "data_new1"
    find_mut_files(str(base_dir))