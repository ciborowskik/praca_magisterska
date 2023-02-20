class MapsWriter:
    def __init__(self, file_path):
        self.file = open(file_path, 'wb')

    def write_next(self, frame):
        self.file.write(frame.flatten())

    def close(self):
        self.file.close()
