'''
Justin Farnsworth
Blobby
November 17, 2020

This is the blob label class, which is a subclass of the pyglet label.
'''

# Imported modules
from pyglet.text import Label


# Blob label class
class BlobLabel(Label):
    # Delete the text from video memory when the object is deleted
    def __del__(self):
        self.delete()
