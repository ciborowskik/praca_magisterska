class YuvWriter:
    def __init__(self, file_path):
        self.file = open(file_path, 'wb')

    def write_next(self, yuv):
        y = yuv[:, :, 0]
        u = yuv[1::2, 1::2, 1]
        v = yuv[1::2, 1::2, 2]

        self.file.write(y.flatten())
        self.file.write(u.flatten())
        self.file.write(v.flatten())

    def close(self):
        self.file.close()
