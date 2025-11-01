import sys
from pathlib import Path

SOURCE = Path(sys.argv[1])
TARGET = Path(sys.argv[2])


def map_path(path: str) -> str:
    if path in {"/dev/null", "dev/null"}:
        return path
    if path.startswith("ggml/"):
        return "ml/backend/ggml/ggml/" + path[len("ggml/"):]
    return "llama/llama.cpp/" + path


def map_line_path(path: str) -> str:
    if path in {"/dev/null", "dev/null"}:
        return path
    if path.startswith("a/"):
        return "a/" + map_path(path[2:])
    if path.startswith("b/"):
        return "b/" + map_path(path[2:])
    return map_path(path)


def transform(lines):
    out_lines = []
    for line in lines:
        if line.startswith("diff --git "):
            parts = line.strip().split()
            if len(parts) >= 4:
                a_path = map_line_path(parts[2])
                b_path = map_line_path(parts[3])
                parts[2] = a_path
                parts[3] = b_path
                out_lines.append(" ".join(parts) + "\n")
            else:
                out_lines.append(line)
        elif line.startswith("--- ") or line.startswith("+++ "):
            prefix = line[:4]
            path = line[4:].strip()
            if path == "/dev/null":
                out_lines.append(prefix + path + "\n")
            else:
                out_lines.append(prefix + map_line_path(path) + "\n")
        elif line.startswith("rename from "):
            path = line[len("rename from "):].strip()
            out_lines.append("rename from " + map_path(path) + "\n")
        elif line.startswith("rename to "):
            path = line[len("rename to "):].strip()
            out_lines.append("rename to " + map_path(path) + "\n")
        elif line.startswith("copy from "):
            path = line[len("copy from "):].strip()
            out_lines.append("copy from " + map_path(path) + "\n")
        elif line.startswith("copy to "):
            path = line[len("copy to "):].strip()
            out_lines.append("copy to " + map_path(path) + "\n")
        else:
            out_lines.append(line)
    return out_lines


def main():
    lines = SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    mapped = transform(lines)
    TARGET.write_text("".join(mapped), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: map_patch.py <source> <target>", file=sys.stderr)
        sys.exit(1)
    main()
