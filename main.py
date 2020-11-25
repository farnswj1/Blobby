'''
Justin Farnsworth
Blobby
November 17, 2020

This is a variation of Agar.io, a massive multiplayer online game in 
which the objective is to eat food and blobs smaller than the player's blob. 
In this implementation, the NEAT algorithm was used to create an AI that will 
control the NPC blobs while the player tries to to grow their own blob. Unlike 
Agar.io, the player will compete against blobs that are controlled by the AI, 
not by other players.

The rules of this game are slightly different than that of Agar.io. For example, 
if any of the blobs, including the player, collides with another blob, the larger 
blob eats the smaller blob and increases its own mass by half the points of the 
smaller blob. However, if the colliding blobs are the same size, both of them die. 
The food supply is constantly regenerated as the blobs eat.

To play the game manually, type the command: `python main.py`

To run the game using NEAT, type the command: `python main.py neat`
'''

# Imported modules
from sys import argv
import Blobby


# Execute the program if called directly
if __name__ == "__main__":
    enable_neat = ("neat" in argv)

    Blobby.run(enable_neat=enable_neat)
