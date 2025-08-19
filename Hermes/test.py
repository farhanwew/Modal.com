import numpy as np
import io, codecs

# bikin array numpy
arr = np.array([1,2,3])

# simpan ke buffer (bukan file)
buffer = io.BytesIO()
np.save(buffer, arr)

# ambil isi buffer (biner)
binary_data = buffer.getvalue()

# encode ke base64
b64_data = codecs.encode(binary_data, "base64")
print(b64_data)


# decode base64 ke biner
decoded_binary = codecs.decode(b64_data, "base64")

# load array langsung dari buffer biner
arr_restored = np.load(io.BytesIO(decoded_binary))
print(arr_restored)   # [1 2 3]
print(type(arr_restored))  # <class 'numpy.ndarray'>

