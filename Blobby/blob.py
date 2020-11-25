'''
Justin Farnsworth
Blobby
November 17, 2020

This is the blob class, which is a subclass of the pyglet circle.
'''

# Imported modules
from pyglet.shapes import Circle
from pyglet.text import Label
from math import pi, sqrt


# Blob class (inherits the pyglet circle class)
class Blob(Circle):
    # Constructor
    def __init__(self, mass, *args, **kwargs):
        # Inherit the pyglet circle
        super().__init__(radius=sqrt(10*mass/pi), *args, **kwargs)

        # Save the mass
        self._mass = mass

        # Generate its velocity on the x-axis and y-axis
        self.velx = 0
        self.vely = 0
    

    # Get the mass
    @property
    def mass(self):
        return self._mass
    

    # Set the mass and update the radius
    @mass.setter
    def mass(self, mass):
        self._mass = mass
        self.radius = sqrt(10*mass/pi)
    

    # Update the blob
    def update(self, dt):
        self.x += self.velx * dt
        self.y += self.vely * dt
