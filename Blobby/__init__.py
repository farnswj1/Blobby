'''
Justin Farnsworth
Blobby
November 17, 2020

This script initializes the Blobby package. To run the game, use the run() method below.
'''

# Imported modules
from .gameWindow import GameWindow


# Run the game
def run(enable_neat=False):
    GameWindow(enable_neat=enable_neat)
