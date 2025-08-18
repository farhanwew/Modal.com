from repeng import ControlVector

# Load vector dari gguf
vec = ControlVector.load_gguf("girly.gguf")

# Simpan ulang ke npz
vec.save("girly.npz")
