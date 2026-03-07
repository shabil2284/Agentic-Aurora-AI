import time
import tempfile

start = time.time()
with open("test_dummy.pdf", "wb") as f:
    f.write(b"dummy pdf" * 1024 * 1024) # 9MB

bytes_data = open("test_dummy.pdf", "rb").read()
print(f"Read bytes: {time.time() - start:.3f}s")

start2 = time.time()
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(bytes_data)
print(f"Write tempfile: {time.time() - start2:.3f}s")
