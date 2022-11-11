# -*- coding: utf-8 -*-
"""
...............................................................................

╔═════════════════════════════════════════════════════════════════════════════╗
║                             Copyright statement                             ║
╠═════════════════════════════════════════════════════════════════════════════╣
  Public
  Copyright by Ben De Jonge
╚═════════════════════════════════════════════════════════════════════════════╝


╔═════════════════════════════════════════════════════════════════════════════╗
║                                  Creation                                   ║
╠═════════════════════════════════════════════════════════════════════════════╣
  Created on Thu Nov 10 21:49:43 2022
  Created by Ben De Jonge
╚═════════════════════════════════════════════════════════════════════════════╝


╔═════════════════════════════════════════════════════════════════════════════╗
║                                   Purpose                                   ║
╠═════════════════════════════════════════════════════════════════════════════╣
  
╚═════════════════════════════════════════════════════════════════════════════╝

...............................................................................
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import imageio as imio
import os

class Body:
    """
    A point-mass body floating in a 2D gravitationless and vaccuum universe.

    Parameters
    ----------
    mass : float
        The mass of the body [unitless].
    position : tuple[int, int]
        The position of the body in Carthesian coordinates.
    momentum : tuple[int, int]
        The momentum vector of the body in Carthesia coordinates.
    """
    
    def __init__(self, 
                 mass : float,
                 position : tuple[int, int],
                 momentum : tuple[int, int]):
        self.mass = mass
        self.x, self.y = self.position = np.array(position, dtype=np.float64)
        self.px, self.py = self.momentum = np.array(momentum, dtype=np.float64)
    
    def get_distance(self, other) -> float:
        """
        Calculate the Euclidian distance between mass centers of two Bodies.

        Parameters
        ----------
        other : Body
            The second Body to calculate the distance to.

        Returns
        -------
        float
            The Euclidian distance between mass centers of two bodies.
        """
        return np.linalg.norm(self.position - other.position)
    
    def get_gravitational_force(self, other, gct : float=9.81) -> float:
        """
        Calculate the magnitude of the gravitational force between two Bodies.

        Parameters
        ----------
        other : Body
            The second Body to calculate the gravitational force between.
        gct : float, optional
            The gravitational constant. The default is 9.81.

        Returns
        -------
        float
            The magnitude of the gravitational force between two Bodies.
        """
        return gct * (self.mass * other.mass)/self.get_distance(other)
    
    def get_delta_momentum(self, other, gct:float=9.81) -> np.ndarray:
        """
        Calculate the change in momentum caused by gravitational pull of a second Body.

        Parameters
        ----------
        other : Body
            The second Body.
        gct : float, optional
            The gravitational constant. The default is 9.81.

        Returns
        -------
        np.ndarray
            The modified momentum vector of the Body.
        """
        return self.get_gravitational_force(other, gct) * (other.position - self.position)
    
    def __repr__(self):
        """Get a string representation of a Body."""
        return f'Body(m={self.mass}, r={self.position}, p={self.momentum})'
        

class Universe:
    """
    A 2D gravitationless and vaccuum universe, containing bodies.

    Parameters
    ----------
    bodies : tuple[Body, Body, Body]
        A set of all bodies in the universe.
    gct : float, optional
        The gravitational constant of the universe. The default is 9.81.
    fps : float, optional
        The frames per second in animation. The default is 10.
    """
    
    def __init__(self,
                 bodies : tuple[Body, Body, Body],
                 gct : float = 9.81,
                 fps : float = 10):
        self.bodies = bodies
        self.gct = gct
        self.fps = fps
    
    def __repr__(self):
        """Get a string representation of a Universe."""
        bodies_repr = '\n'.join(b.__repr__() for b in self.bodies)
        return f'Universe with bodies:\n{bodies_repr}'
    
    def get_com(self) -> np.ndarray:
        """
        Calculate the center-of-mass of the Universe.

        Returns
        -------
        com : np.ndarray
            The coordinates of the center-of-mass.
        """
        mass_total = sum(body.mass for body in self.bodies)
        com = sum(b.mass * b.position for b in self.bodies) / mass_total
        return com
    
    def move(self) -> tuple[Body, Body, Body]:
        """
        Update the position and momentum vectors of all Bodies.

        Returns
        -------
        tuple[Body, Body, Body]
            Tuple of the original Bodies with updated position and momentum vectors.
        """
        # Containers for updated parameters.
        new_positions = []
        new_momenta   = []
        for body in self.bodies:
            # Update position based on current momentum.
            new_positions.append(body.position + body.momentum)
            # Update momentum based on other bodies.
            new_momentum = body.momentum
            others = list(self.bodies)
            others.remove(body)
            for other in others:
                new_momentum += body.get_delta_momentum(other, gct=self.gct)
            new_momenta.append(new_momentum)
        # Store updated parameters in bodies.
        for body, position, momentum in zip(self.bodies, new_positions, new_momenta):
            body.position = position
            body.momentum = momentum
        return self.bodies
    
    def create_frame(self, path : str, step : int=0, plot_trail : bool=False,
                     previous : tuple[deque, deque]=None, 
                     box : int=40_000, trail_frames:int=150) -> (deque, deque):
        """
        Plot the current situation of the universe.

        Parameters
        ----------
        path : str
            The output filename of the plot.
        step : int, optional
            The current step number in the animation. The default is 0.
        plot_trail : bool, optional
            Whether (True) or not (False) to animate a trail. Note this strongly
            increases the required rendering time. The default is False.
        previous : tuple[deque, deque], optional
            Previous body x and y positions to plot a trail. The default is None.
        box : int, optional
            Field-of-view around the center-of-mass. The default is 40_000.
        trail_frames : int, optional
            The number of frames the trail remains visible during animation.
            The default is 150.

        Returns
        -------
        trail_x : deque
            A deque of previous x-positions of the bodies.
        trail_y : deque
            A deque of previous y-positions of the bodies.
        """
        # Creating Figure to plot on.
        fig = plt.figure(figsize=(6,6))
        colors = ['red', 'green', 'blue']
        # Plotting new positions.
        xs = list(b.position[0] for b in self.bodies)
        ys = list(b.position[1] for b in self.bodies)
        plt.scatter(xs, ys, color=colors, zorder = 2)
        # Plotting trail if applicable.
        trail_x, trail_y = None, None
        if plot_trail:
            if not previous:
                trail_x = deque(maxlen=trail_frames)
                trail_y = deque(maxlen=trail_frames)
            else:
                trail_x, trail_y = previous
            trail_x.append(xs)
            trail_y.append(ys)
            for i, (tx, ty) in enumerate(zip(trail_x, trail_y)):
                plt.scatter(tx, ty, marker='.', color=colors, zorder=1, alpha=(i)/(5*len(trail_x)))
        # Plotting center-of-mass.
        com = self.get_com()
        plt.scatter(com[0], com[1], marker='.', color='black', zorder=0)
        # Formatting plot around center-of-mass.
        plt.xlim((-box + com[0], com[0] + box))
        plt.ylim((-box + com[1], com[1] + box))
        plt.title(f'Timestep {step}')
        # Saving and closing.
        plt.savefig(path,
                    transparent=False, facecolor='white')
        plt.close()
        return trail_x, trail_y
    
    def animate(self, path: str='./animations', plot_trail: bool=False, 
                duration: int=3000, fps: int=30):
        """
        Create an animation of the universe over a given time period.

        Parameters
        ----------
        path : str, optional
            The destination folder for the animation. The default is './animations'.
        plot_trail : bool, optional
            Whether (True) or not (False) to animate a trail. Note this strongly
            increases the required rendering time. The default is False.
        duration : int, optional
            The number of frames to animate. The default is 3000.
        fps : int, optional
            The framerate in frames per second. The default is 30.

        Returns
        -------
        None.
        """
        frames = []
        previous = None
        for step in range(duration):
            if step % 10 == 0:
                print(f'Animating step {step}/{duration}.')
            file = os.path.join(path, f'im_{step}.png')
            previous = self.create_frame(path=file, step=step, 
                                         previous=previous, plot_trail=plot_trail)
            self.bodies = self.move()
            frames.append(imio.v2.imread(path))
            os.remove(path)
        imio.mimsave(uri='./animations/animation.gif', ims=frames, fps=fps)

if __name__ == '__main__':
    b1 = Body(1, (1, 2), (2,-5))
    b2 = Body(1, (4,4), (1,3))
    b3 = Body(1, (-4,-6), (5,2))

    universe = Universe(bodies = (b1, b2, b3))
    universe.animate(plot_trail=True, duration=3000, fps=30)