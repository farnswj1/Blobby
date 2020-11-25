'''
Justin Farnsworth
Blobby
November 17, 2020

The game event handler class handles the game objects and logic.
'''

# Imported modules
import pyglet
import neat
import pickle
import os
from pyglet.window import Window, key
from pyglet.text import Label
from .blob import Blob
from .food import Food
from random import randint, randrange
from math import sqrt, pi
from itertools import combinations


# Game window class
class GameEventHandler:
    # Constructor
    def __init__(self, width, height, enable_neat=False, *args, **kwargs):
        # Save the configuration that determines if NEAT is enabled
        self.enable_neat = enable_neat

        # Save the window dimensions
        self.width = width
        self.height = height

        # Keep track of when the program is terminated
        self.user_exit = False

        # Batch objects
        self.blob_batch = pyglet.graphics.Batch()
        self.label_batch = pyglet.graphics.Batch()
        self.hud_batch = pyglet.graphics.Batch()
        if not self.enable_neat:
            self.game_over_batch = pyglet.graphics.Batch()

        # Number of blobs (overridden if NEAT is enabled)
        self.number_of_blobs = 16

        # Number of food items
        self.number_of_foods = 300

        # Score
        self.score = 50
        self.score_label = Label(
            f"Score: {self.score}",
            font_name="Times New Roman",
            font_size=20,
            color=(0, 0, 0, 255),
            x=self.width - 10,
            y=self.height - 10,
            anchor_x="right",
            anchor_y="top",
            batch=self.hud_batch
        )

        # Lists of objects
        self.player = None
        self.blobs = []
        self.blob_labels = []
        self.neural_nets = []
        self.genomes = []
        self.foods = []

        # Locate the NEAT configuration file
        config_file = os.path.join(os.path.dirname(__file__), "neat_config.txt")

        # Configure the NEAT algorithm
        self.neat_config = neat.config.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file
        )

        # If NEAT is disabled, set up the game for the user to play manually
        if not self.enable_neat:
            # Generate the player's blob
            self.player = Blob(
                x=self.width/2,
                y=self.height/2,
                mass=50,
                segments=50,
                color=(0, 128, 255),
                batch=self.blob_batch
            )

            # Generate the player's associated label
            self.player_label = Label(
                f"{self.player.mass}",
                font_name="Times New Roman",
                font_size=self.player.radius * 0.6,
                x=self.player.x,
                y=self.player.y,
                anchor_x="center",
                anchor_y="center",
                batch=self.label_batch
            )

            # Keep track of player inputs
            self.player_moving_left = False
            self.player_moving_right = False
            self.player_moving_up = False
            self.player_moving_down = False

            # Blob respawn timer
            self.next_blob_spawn = 5

            # Load the AI into the game
            pkl_file = "genome.pkl"
            if os.path.exists(pkl_file):
                with open(pkl_file, "rb") as f:
                    self.genome = pickle.load(f)

            # Generate the blob NPCs and make sure none of them overlap
            for _ in range(self.number_of_blobs):
                self.generate_blob(
                    neural_net=neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config),
                    genome=self.genome,
                    mass=self.score
                )

            # Generate the food particles and ensure they don't overlap with anything
            for _ in range(self.number_of_foods):
                self.generate_food()
            
            # Game Over label
            self.game_over_label = Label(
                "Game Over",
                font_name="Times New Roman",
                font_size=self.width / 10,
                x=self.width / 2,
                y=self.height * 2 / 3,
                anchor_x="center",
                anchor_y="center",
                color=(0, 0, 0, 255),
                batch=self.game_over_batch
            )

            # Reset label
            self.reset_label = Label(
                "Press ENTER to reset the game.",
                font_name="Times New Roman",
                font_size=self.width / 25,
                x=self.width / 2,
                y=self.height / 3,
                anchor_x="center",
                anchor_y="center",
                color=(0, 0, 0, 255),
                batch=self.game_over_batch
            )

            # Exit label
            self.exit_label = Label(
                "Press ESC to exit the game.",
                font_name="Times New Roman",
                font_size=self.width / 25,
                x=self.width / 2,
                y=self.height / 6,
                anchor_x="center",
                anchor_y="center",
                color=(0, 0, 0, 255),
                batch=self.game_over_batch
            )
    

    # Setup and run the game with the NEAT algorithm
    def run_neat(self):
        # Generate the population
        population = neat.Population(self.neat_config)

        # Add a reporter to show progress in the terminal
        population.add_reporter(neat.StdOutReporter(True))
        population.add_reporter(neat.StatisticsReporter())

        # Generation label
        self.generation = -1
        self.generation_label = Label(
            f"Generation: {self.generation}",
            font_name="Times New Roman",
            font_size=20,
            color=(0, 0, 0, 255),
            x=10,
            y=self.height - 10,
            anchor_x="left",
            anchor_y="top",
            batch=self.hud_batch
        )

        # Number of blobs label
        self.number_of_blobs_label = Label(
            f"Blobs: {self.number_of_blobs}",
            font_name="Times New Roman",
            font_size=20,
            color=(0, 0, 0, 255),
            x=10,
            y=self.height - 40,
            anchor_x="left",
            anchor_y="top",
            batch=self.hud_batch
        )

        # Run the NEAT algorithm and find the best AI
        winner = population.run(self.eval_genomes, 100)

        # If the program is terminated at the last generation, don't show the results
        if not self.has_exit:
            # Print the genome that performed the best
            print(f"\nBest genome:\n{winner}")

            # If the .pkl file exists, save the best genome
            pkl_file = "genome.pkl"
            if os.path.exists(pkl_file):
                # Get the current genome saved in the .pkl file
                with open(pkl_file, "rb") as f:
                    saved_genome = pickle.load(f)
                        
                # Save the winner genome if it has a higher fitness than the saved genome
                if winner.fitness > saved_genome.fitness:
                    print(
                        "This genome performed better than the saved genome!\n"
                        "Overwriting the saved genome with this session's best genome..."
                    )
                    with open(pkl_file, "wb") as f:
                        pickle.dump(winner, f)
            else:
                # Create the .pkl file and save the winner genome
                print("Generating genome.pkl and saving the best genome...")
                with open(pkl_file, "wb") as f:
                    pickle.dump(winner, f)
    

    # Generate a blob NPC
    def generate_blob(self, genome, neural_net, mass=None):
        # Generate a random mass if specified
        random_mass = mass + randint(-25, 50) if mass else 50

        # Keep searching for a valid coordinate to place the blob
        blob_created = False
        while not blob_created:
            # Generate random coordinates
            x = randint(50, self.width - 51)
            y = randint(50, self.height - 51)

            # Keep track of the validity of the coordinates
            valid_coordinates = True

            # Check if the coordinates are too close to the player 
            # if NEAT is disabled and if it is alive or exists
            if not self.enable_neat:
                if self.player:
                    if self.distance(self.player.x, self.player.y, x, y) < 50:
                        valid_coordinates = False
             
            # Check all the blobs to ensure the coordinates aren't too close to them
            for blob in self.blobs:
                if self.distance(blob.x, blob.y, x, y) < 50:
                    valid_coordinates = False
                
            # Append the blob if the coordinates are valid
            if valid_coordinates:
                self.blobs.append(
                    Blob(
                        x=x,
                        y=y,
                        mass=random_mass,
                        segments=50,
                        color=(255, 0, 0),
                        batch=self.blob_batch
                    )
                )

                # Create its associated label
                self.blob_labels.append(
                    Label(
                        f"{random_mass}",
                        font_name="Times New Roman",
                        font_size=sqrt(10 * random_mass / pi) * 0.6,
                        x=x,
                        y=y,
                        anchor_x="center",
                        anchor_y="center",
                        batch=self.label_batch
                    )
                )

                # Create the genome and its neural network
                self.neural_nets.append(neural_net)
                self.genomes.append(genome)

                # Terminate the while loop
                blob_created = True
    

    # Generate a food item
    def generate_food(self):
        # Keep searching for a valid coordinate to place the food item
        food_created = False
        while not food_created:
            # Generate random coordinates
            x = randint(50, self.width - 51)
            y = randint(50, self.height - 51)

            # Keep track of the validity of the coordinates
            valid_coordinates = True

            # Check if the coordinates are too close to the player 
            # if NEAT is disabled and if it is alive or exists
            if not self.enable_neat:
                if self.player:
                    if self.distance(self.player.x, self.player.y, x, y) < 25:
                        valid_coordinates = False
             
            # Check all the blobs to ensure the coordinates aren't too close to them
            for blob in self.blobs:
                if self.distance(blob.x, blob.y, x, y) < 25:
                    valid_coordinates = False
            
            # Append the food item if the coordinates are valid
            if valid_coordinates:
                self.foods.append(
                    Food(
                        x=x,
                        y=y,
                        width=4,
                        height=4,
                        color=tuple(randrange(256) for _ in range(3)),
                        batch=self.blob_batch
                    )
                )

                # Terminate the while loop
                food_created = True
    

    # Get the distance between two points
    def distance(self, x1, y1, x2, y2):
        return sqrt((x2-x1)**2 + (y2-y1)**2)


    # Draw the contents on the screen
    def draw(self):
        self.blob_batch.draw()
        self.label_batch.draw()
        self.hud_batch.draw()

        # Only show if NEAT is disabled and the player dies
        if not self.enable_neat and not self.player:
            self.game_over_batch.draw()
    

    # Keep the blob completely on the screen
    def adjust_blob_position(self, blob):
        # Adjust horizontal position if necessary
        if blob.x < self.width / 2:
            blob.x = max(blob.x, blob.radius)
        else:
            blob.x = min(blob.x, self.width - 1 - blob.radius)

        # Adjust vertical position if necessary
        if blob.y < self.height / 2:
            blob.y = max(blob.y, blob.radius)
        else:
            blob.y = min(blob.y, self.height - 1 - blob.radius)
    

    # Update the player
    def update_player(self, dt):
        # Change the net velocities depending the key(s) pressed
        net_velx = 0
        net_vely = 0

        # Change the net velocity by a specified amount
        increment = 100

        # If the player chooses to go left, decrease the net horizontal velocity
        if self.player_moving_left and self.player.x - self.player.radius >= 0:
            net_velx -= increment
        
        # If the player chooses to go right, increase the net horizontal velocity
        if self.player_moving_right and self.player.x + self.player.radius < self.width:
            net_velx += increment

        # Apply the net horizontal velocity to the player
        self.player.velx = net_velx

        # If the player chooses to go down, increase the net vertical velocity
        if self.player_moving_down and self.player.y - self.player.radius >= 0:
            net_vely -= increment
        
        # If the player chooses to go up, decrease the net vertical velocity
        if self.player_moving_up and self.player.y + self.player.radius < self.height:
            net_vely += increment
        
        # Apply the net vertical velocity to the player
        self.player.vely = net_vely

        # Update the player's position
        self.player.update(dt)
        self.player_label.x = self.player.x
        self.player_label.y = self.player.y
    

    # Update the window and the objects
    def update(self, dt):
        # Add to the set indices of blobs that have been killed
        dead_blobs = set()
        dead_blob_labels = set()
        dead_neural_nets = set()
        dead_genomes = set()

        # If the player exists, check for collisions between the player and the other blobs
        if self.player:
            for blob, label, neural_net, genome in zip(self.blobs, self.blob_labels, self.neural_nets, self.genomes):
                if self.player and self.distance(self.player.x, self.player.y, blob.x, blob.y) < self.player.radius + blob.radius:
                    # The largest of the two will consume the other.
                    # If they're the same size, delete both of them 
                    if self.player.mass > blob.mass:
                        # Update the player's mass and label
                        self.player.mass += blob.mass // 2
                        self.player_label.text = f"{self.player.mass}"
                        self.player_label.font_size = self.player.radius * 0.6
                        
                        # Update the score
                        self.score = self.player.mass
                        self.score_label.text = f"Score: {self.score}"

                        # Make sure the player remains completely on the screen
                        self.adjust_blob_position(self.player)

                        # Add the blob to the set of dead blobs
                        dead_blobs.add(blob)
                        dead_blob_labels.add(label)
                        dead_neural_nets.add(neural_net)
                        dead_genomes.add(genome)
                    elif self.player.mass < blob.mass:
                        # Update the player's mass and label
                        blob.mass += self.player.mass // 2
                        label.text = f"{blob.mass}"
                        label.font_size = blob.radius * 0.6
                        genome.kill_timer = 10 # Reset the kill timer

                        # Make sure the blob remains completely on the screen
                        self.adjust_blob_position(blob)

                        # Delete the player and the associated label
                        # Python GC will handle the objects once the reference is removed
                        self.player_label.delete() # Delete the label from video memory
                        self.player = None 
                        self.player_label = None
                    else:
                        # Delete the blob
                        dead_blobs.add(blob)
                        dead_blob_labels.add(label)
                        dead_neural_nets.add(neural_net)
                        dead_genomes.add(genome)

                        # Delete the player and the associated label
                        # Python GC will handle the objects once the reference is removed
                        self.player_label.delete() # Delete the label from video memory
                        self.player = None 
                        self.player_label = None
        
        # Check for collisions among the blobs
        for blob_1, blob_2 in combinations(self.blobs, 2):
            if self.distance(blob_1.x, blob_1.y, blob_2.x, blob_2.y) < blob_1.radius + blob_2.radius:
                # Get the indices of both blobs
                blob_1_index = self.blobs.index(blob_1)
                blob_2_index = self.blobs.index(blob_2)

                # The largest of the two will consume the other.
                # If they're the same size, delete both of them
                if blob_1.mass > blob_2.mass:
                    # Update the larger blob's mass and its associated label
                    blob_1.mass += blob_2.mass // 2
                    self.blob_labels[blob_1_index].text = f"{blob_1.mass}"
                    self.blob_labels[blob_1_index].font_size = blob_1.radius * 0.6
                    self.genomes[blob_1_index].kill_timer = 10 # Reset the kill timer

                    # Make sure the larger blob remains completely on the screen
                    self.adjust_blob_position(blob_1)

                    # Add the smaller blob's index to the set of dead blobs
                    dead_blobs.add(self.blobs[blob_2_index])
                    dead_blob_labels.add(self.blob_labels[blob_2_index])
                    dead_neural_nets.add(self.neural_nets[blob_2_index])
                    dead_genomes.add(self.genomes[blob_2_index])
                elif blob_1.mass < blob_2.mass:
                    # Update the blob's mass and its associated label
                    blob_2.mass += blob_1.mass // 2
                    self.blob_labels[blob_2_index].text = f"{blob_2.mass}"
                    self.blob_labels[blob_2_index].font_size = blob_2.radius * 0.6
                    self.genomes[blob_2_index].kill_timer = 10 # Reset the kill timer

                    # Make sure the larger blob remains completely on the screen
                    self.adjust_blob_position(blob_2)

                    # Add the smaller blob's index to the set of dead blobs
                    dead_blobs.add(self.blobs[blob_1_index])
                    dead_blob_labels.add(self.blob_labels[blob_1_index])
                    dead_neural_nets.add(self.neural_nets[blob_1_index])
                    dead_genomes.add(self.genomes[blob_1_index])
                else:
                    # Add both blobs' indices to the set of dead blobs
                    dead_blobs.add(self.blobs[blob_1_index])
                    dead_blobs.add(self.blobs[blob_2_index])
                    dead_blob_labels.add(self.blob_labels[blob_1_index])
                    dead_blob_labels.add(self.blob_labels[blob_2_index])
                    dead_neural_nets.add(self.neural_nets[blob_1_index])
                    dead_neural_nets.add(self.neural_nets[blob_2_index])
                    dead_genomes.add(self.genomes[blob_1_index])
                    dead_genomes.add(self.genomes[blob_2_index])
        
        # If NEAT is enabled, decrement the kill timer. If any reach 0, delete the blob genome.
        # Also check for blobs that collided with the window border
        if self.enable_neat:
            for blob, label, neural_net, genome in zip(self.blobs, self.blob_labels, self.neural_nets, self. genomes):
                if genome.kill_timer <= 0 or (
                blob.x - blob.radius <= 0 or blob.x + blob.radius >= self.width - 1
                or blob.y - blob.radius <= 0 or blob.y + blob.radius >= self.height - 1):
                    # Penalize the genome heavily
                    genome.fitness -= 200

                    # Add the blob genome to the list of dead blobs
                    dead_blobs.add(blob)
                    dead_blob_labels.add(label)
                    dead_neural_nets.add(neural_net)
                    dead_genomes.add(genome)
                else:
                    genome.kill_timer -= dt
        
        # Remove the dead objects from the game
        for blob, label, neural_net, genome in zip(dead_blobs, dead_blob_labels, dead_neural_nets, dead_genomes):
            # Penalize the genome for dying
            genome.fitness -= 100

            # Delete the label from video memory
            label.delete()
            
            # Eliminate the genome
            self.blobs.remove(blob)
            self.blob_labels.remove(label)
            self.neural_nets.remove(neural_net)
            self.genomes.remove(genome)

            # Update the number of blobs remaining if NEAT is enabled
            if self.enable_neat:
                self.number_of_blobs_label.text = f"Blobs: {len(self.blobs)}"
            
                # If no blobs remain, terminate the current generation
                if not self.blobs:
                    self.reset()
                    pyglet.app.exit()
        
        # Check for collisions between the player and the food if the player is alive
        if self.player:
            for food in self.foods:
                if self.distance(self.player.x, self.player.y, food.x, food.y) < self.player.radius + food.width / 2:
                    # Update the player and the associated label
                    self.player.mass += 1
                    self.player_label.text = f"{self.player.mass}"
                    self.player_label.font_size = self.player.radius * 0.6
                    
                    # Update the score
                    self.score = self.player.mass
                    self.score_label.text = f"Score: {self.score}"

                    # Make sure the player remains completely on the screen
                    self.adjust_blob_position(self.player)

                    # Delete the current food item and generate a new one
                    self.foods.remove(food)
                    self.generate_food()

        # Check for collisions between the blob NPCs and the food
        for blob, label, genome in zip(self.blobs, self.blob_labels, self.genomes):    
            for food in self.foods:
                # Keep checking the blobs until we find that a blob ate the food item
                food_consumed = False
                if not food_consumed and self.distance(blob.x, blob.y, food.x, food.y) < blob.radius + food.width:
                    # Update the blob's mass and its associated label
                    blob.mass += 1
                    label.text = f"{blob.mass}"
                    label.font_size = blob.radius * 0.6

                    # Reset the kill timer
                    genome.kill_timer = 10

                    # Make sure the blob remains completely on the screen
                    self.adjust_blob_position(blob)

                    # Delete the current food item and generate a new one
                    self.foods.remove(food)
                    self.generate_food()

                    # Reward the genome for finding food
                    genome.fitness += 1

                    # Terminate the inner loop
                    food_consumed = True
            
        # Let the neural networks make decisions for their respective blobs
        for blob, neural_net, genome in zip(self.blobs, self.neural_nets, self.genomes):
            # Get the closest blob
            closest_blob = None
            min_blob_distance = float("Inf")
            for other_blob in self.blobs:
                distance = self.distance(blob.x, blob.y, other_blob.x, other_blob.y)
                if blob is not other_blob and distance < min_blob_distance:
                    closest_blob = other_blob
                    min_blob_distance = distance
            
            # If the player exists and is the closest blob, save it as the closest blob.
            # Otherwise, save the the current blob as the closest blob if no other blob exists
            if self.player: # The player exists
                if closest_blob: # At least one other blob exists
                    distance = self.distance(self.player.x, self.player.y, closest_blob.x, closest_blob.y)
                    if distance < min_blob_distance:
                        closest_blob = self.player
                        min_blob_distance = distance
                else: # There is only one blob NPC alive
                    closest_blob = self.player
                    min_blob_distance = distance
            else:
                if not closest_blob: # No other blob NPC exists
                    closest_blob = blob
            
            # If there is a closest blob, extract the needed attributes.
            # Otherwise, set some values
            closest_blob_x = closest_blob.x if closest_blob else self.width / 2
            closest_blob_y = closest_blob.y if closest_blob else self.height / 2
            closest_blob_mass = closest_blob.mass if closest_blob else 0
                        
            # Get the closest food item
            closest_food = None
            min_food_distance = float("Inf")
            for food in self.foods:
                distance = self.distance(blob.x, blob.y, food.x, food.y)
                if distance < min_food_distance:
                    closest_food = food
                    min_food_distance = distance
                    
            # Activate the genome's neural network which will determine the blob's next move
            output = neural_net.activate((
                blob.x, # X-coordinate of the blob
                blob.y, # Y-coordinate of the blob
                #blob.x - blob.radius, # Distance from the left window border
                #self.width - blob.x - blob.radius, # Distance from the right window border
                #blob.y - blob.radius, # Distance from the top window border
                #self.height - blob.y - blob.radius, # Distance from the bottom window border
                blob.x - closest_food.x, # Distance to the food item on the x-axis
                blob.y - closest_food.y, # Distance to the food item on the y-axis
                blob.x - closest_blob_x if closest_blob else self.width / 2, # Distance to the blob on the x-axis
                blob.y - closest_blob_y if closest_blob else self.height / 2, # Distance to the blob on the y-axis
                blob.mass - closest_blob_mass if closest_blob else 0 # Difference between the masses of the two blobs
            ))

            # Change the velocity by a specified amount
            increment = 100

            # Horizontal movement of the blob
            if output[0] < -0.5 and blob.x - blob.radius >= 0: # Left
                blob.velx = -increment
            elif output[0] > 0.5 and blob.x + blob.radius < self.width: # Right
                blob.velx = increment
            else: # No movement
                blob.velx = 0
            
            # Vertical movement of the blob
            if output[1] < -0.5 and blob.y - blob.radius >= 0: # Down
                blob.vely = -increment
            elif output[1] > 0.5 and blob.y + blob.radius < self.height: # Up
                blob.vely = increment
            else: # No movement
                blob.vely = 0
        
        # Update the player's position if the player is alive
        if self.player:
            self.update_player(dt)
        
        # Update the position of the blobs and their labels.
        # Also if NEAT is enabled, update the score so that it's the mass of the largest blob
        for blob, label in zip(self.blobs, self.blob_labels):
            blob.update(dt)
            label.x = blob.x
            label.y = blob.y

            # If NEAT is enabled, update the score
            if self.enable_neat and blob.mass > self.score:
                self.score = blob.mass
                self.score_label.text = f"Score: {self.score}"
        
        # If the list is short and NEAT is disabled, update the blob respawn timer.
        # If the time is up, spawn a blob
        if not self.enable_neat and len(self.blobs) < self.number_of_blobs:
            if self.next_blob_spawn <= 0:
                self.generate_blob(
                    neural_net=neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config),
                    genome=self.genome,
                    mass=self.score
                )
                self.next_blob_spawn = 5
            else:
                self.next_blob_spawn -= dt
    

    # Run the game with the NEAT algorithm
    def eval_genomes(self, genomes, config):
        # Terminate if the user closed the window
        if self.user_exit:
            exit()

        # Increment the generation number
        self.generation += 1
        self.generation_label.text = f"Generation: {self.generation}"
        self.number_of_blobs = len(genomes)
        self.number_of_blobs_label.text = f"Blobs: {self.number_of_blobs}"

        # Reset the list of genomes
        self.blobs.clear()
        self.neural_nets.clear()
        self.genomes.clear()

        # Set up the genomes
        for _, genome in genomes:
            genome.fitness = 0  # Start with fitness level of 0
            genome.kill_timer = 10 # Genome has at least 10 seconds to eat something
            self.generate_blob(
                neural_net=neat.nn.FeedForwardNetwork.create(genome, self.neat_config),
                genome=genome
            )

        # Generate the food particles and ensure they don't overlap with anything
        for _ in range(self.number_of_foods):
            self.generate_food()
        
        # Run the game
        pyglet.app.run()
    

    # Reset the game
    def reset(self):        
        # Clear the list of blobs and food items
        self.blobs.clear()
        self.foods.clear()
        
        # Delete the images from video memory, then clear the labels list
        for label in self.blob_labels:
            label.delete()
        self.blob_labels.clear()

        # Reset score label
        self.score = 50
        self.score_label.text = f"Score: {self.score}"

        # Reset the player and generate the objects here if NEAT is disabled.
        # If NEAT is enabled, then the objects are generated at eval_genomes()
        if not self.enable_neat:
            # Reset the player
            self.player = Blob(
                x=self.width/2,
                y=self.height/2,
                mass=50,
                segments=50,
                color=(0, 128, 255),
                batch=self.blob_batch
            )

            # Reset the player's associated label
            self.player_label = Label(
                f"{self.player.mass}",
                font_name="Times New Roman",
                font_size=self.player.radius * 0.6,
                x=self.player.x,
                y=self.player.y,
                anchor_x="center",
                anchor_y="center",
                batch=self.label_batch
            )
            
            # Generate the blobs
            for _ in range(self.number_of_blobs):
                self.generate_blob(
                    neural_net=neat.nn.FeedForwardNetwork.create(self.genome, self.neat_config),
                    genome=self.genome,
                    mass=self.score
                )
            
            # Generate the food
            for _ in range(self.number_of_foods):
                self.generate_food()
