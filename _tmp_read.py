import sys, os
os.chdir(r'c:\neoterminal')
fname = sys.argv[1]
start = int(sys.argv[2])
end = int(sys.argv[3])
with open(fname, encoding='utf-8') as f:
    lines = f.read().splitlines()
for i in range(start-1, min(end, len(lines))):
    sys.stdout.write(f"{i+1}|{lines[i]}\n")
