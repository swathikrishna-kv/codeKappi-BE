import os
import glob

def main():
    directory = 'data'
    types = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
    files = []
    for a_type in types:
        files.extend(glob.glob(os.path.join(directory, 'images', a_type)))

if __name__ == '__main__':
    main()