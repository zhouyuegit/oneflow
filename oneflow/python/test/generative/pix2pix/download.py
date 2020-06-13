import tensorflow as tf
import os

def download():
    # the default download path is "~/.keras/datasets"
    if not os.path.exists("data"):
        os.mkdir("data")
    _PATH = os.path.join(os.getcwd(), "data/facades.tar.gz")
    _URL = 'https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz'
    path_to_zip = tf.keras.utils.get_file(_PATH,
                                          origin=_URL,
                                          extract=True)
    return path_to_zip

if __name__ == "__main__":
    download()
