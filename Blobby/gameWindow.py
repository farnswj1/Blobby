'''
Justin Farnsworth
Google Chrome Dinosaur Game (with NEAT)
November 17, 2020

This is the game window, which is a subclass of the window in pyglet.
'''

# Imported modules
import pyglet
from pyglet.window import Window, key
from .gameEventHandler import GameEventHandler


# Game window class
class GameWindow(Window):
    # Constructor
    def __init__(self, enable_neat=False, *args, **kwargs):
        # Inherit the pyglet window
        super().__init__(
            caption="Blobby",
            fullscreen=True,
            *args,
            **kwargs
        )

        # Save the configuration that determines if NEAT is enabled
        self.enable_neat = enable_neat

        # Set the background to faux-pale lavender
        pyglet.gl.glClearColor(0.9, 0.9, 1, 1)

        # Set the FPS
        pyglet.clock.schedule_interval(self.update, 1/60)

        # Create the game instance
        self.game = GameEventHandler(width=self.width, height=self.height, enable_neat=enable_neat)

        # If NEAT is enabled, run the game using NEAT. Otherwise, let the player play manually
        if enable_neat:
            self.game.run_neat()
        else:
            pyglet.app.run()
    

    # Handle the events when a key is pressed
    def on_key_press(self, symbol, modifiers):
        # Terminate the game if the ESC is pressed
        if symbol == key.ESCAPE:
            self.on_close()
        
        # Check for the following key presses if NEAT is disabled
        if not self.enable_neat:
            # Move left
            if symbol in (key.LEFT, key.A):
                self.game.player_moving_left = True
            
            # Move right
            if symbol in (key.RIGHT, key.D):
                self.game.player_moving_right = True
            
            # Move down
            if symbol in (key.DOWN, key.S):
                self.game.player_moving_down = True
            
            # Move up
            if symbol in (key.UP, key.W):
                self.game.player_moving_up = True
            
            # Reset option (available if the player dies)
            if not self.game.player and symbol == key.ENTER:
                self.game.reset()

    
    # Handle the events when a key is released
    def on_key_release(self, symbol, modifiers):
        # Check for the following key presses if NEAT is disabled
        if not self.enable_neat:
            # Move left
            if symbol in (key.LEFT, key.A):
                self.game.player_moving_left = False
            
            # Move right
            if symbol in (key.RIGHT, key.D):
                self.game.player_moving_right = False
            
            # Move down
            if symbol in (key.DOWN, key.S):
                self.game.player_moving_down = False
            
            # Move up
            if symbol in (key.UP, key.W):
                self.game.player_moving_up = False


    # Draw the contents on the screen
    def on_draw(self):
        self.clear()
        self.game.draw()
    

    # Update the game
    def update(self, dt):
        self.game.update(dt)
    

    # Terminate the game if the window is closed
    def on_close(self):
        self.game.user_exit = True
        super().on_close()
