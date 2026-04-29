python3 - << 'EOF'
path = "profile_output/20260427_201018/raw/callgrind.out"

with open(path) as f:
    lines = f.readlines()

# Find main in sim-test
for i, line in enumerate(lines):
    if "sim-test" in line and line.startswith("fl="):
        for j in range(i, min(i+120, len(lines))):
            if lines[j].strip():
                print(f"{j:6d}: {lines[j]}", end="")
        break
EOF