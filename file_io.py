# File I/O demo
with open("sample.txt", "w") as f:
    f.write("This is a test.\nLine two here.")

with open("sample.txt", "r") as f:
    content = f.read()
    print(content)
