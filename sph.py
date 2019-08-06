import math
import numpy as np

GRAVITY_ACCELERATION = 9.8067


def kernel_poly_6(h):
    C = 315 / (64 * math.pi * h ** 9)
    def kernel(r):
        r_sq = np.sum(r ** 2, axis=-1)
        return np.where(r_sq < 1, C * (h ** 2 - r_sq) ** 3, np.zeros_like(r_sq)) 

def kernel_gradient_poly_6(h):
    C = 945 / (32 * math.pi * h ** 9)
    def kernel(r):
        r_sq = np.sum(r ** 2, axis=-1)
        return np.where(r_sq < 1, C * -r * (h ** 2 - r_sq) ** 2, np.zeros_like(r)) 

def kernel_laplacian_poly_6(h):
    C = 945 / (8 * math.pi * h ** 9)
    def kernel(r):
        r_sq = np.sum(r ** 2, axis=-1)
        return np.where(r_sq < 1, C * (h ** 2 - r_sq), np.zeros_like(r_sq)) 

def kernel_spiky(h):
    C = 15 / (math.pi * h ** 6)
    def kernel(r):
        r_ab = np.sqrt(np.sum(r ** 2, axis=-1))
        return np.where(r_ab < 1, C * (h - r_ab) ** 3, np.zeros_like(r_ab))

def kernel_gradient_spiky(h):
    C = 45 / (math.pi * h ** 6)
    def kernel(r):
        r_ab = np.sqrt(np.sum(r ** 2, axis=-1))
        return np.where(r_ab < 1, C / r_ab * -r * (h - r_ab) ** 2, np.zeros_like(r_ab))
        
def kernel_viscosity(h):
    C = 15 / (2 * math.pi * h ** 3)
    def kernel(r):
        r_ab = np.sqrt(np.sum(r ** 2, axis=-1))
        return np.where(r_ab < 1, C * ((-r_ab ** 3 / (2 * h ** 3)) + (r_ab ** 2 / h ** 2) + (h / (2 * r_ab))), np.zeros_like(r_ab))

def kernel_gradient_viscosity(h):
    C = 15 / (2 * math.pi * h ** 3)
    def kernel(r):
        r_ab = np.sqrt(np.sum(r ** 2, axis=-1))
        return np.where(r_ab < 1, C * ((-3 * r_ab / (2 * h ** 3)) + (2 / h ** 2) - (h / (2 + r_ab ** 3))), np.zeros_like(r))

def kernel_laplacian_viscosity(h):
    C = 45 / (math.pi * h ** 5)
    def kernel(r):
        r_ab = np.sqrt(np.sum(r ** 2, axis=-1))
        return np.where(r_ab < 1, C * (1 - r_ab / h), np.zeros_like(r_ab))


class ParticleInterface:
    h = 0.0457
    kernel_poly_6 = kernel_poly_6(h)
    kernel_gradient_poly_6 = kernel_gradient_poly_6(h)
    kernel_laplacian_poly_6 = kernel_laplacian_poly_6(h)
    kernel_spiky = kernel_spiky(h)
    kernel_gradient_spiky = kernel_gradient_spiky(h)
    kernel_viscosity = kernel_viscosity(h)
    kernel_gradient_viscosity = kernel_gradient_viscosity(h)
    kernel_laplacian_viscosity = kernel_laplacian_viscosity(h)


class Particle(ParticleInterface):
    def __init__(self):
        self.density = 0
        self.position = np.array((0, 0))
        self.velocity = np.array((0, 0))
        self.mass = 0.02
        self.rest_density = 998.29
        self.viscosity = 3.5
        self.gas_stiffness = 3.0
        self.surface_tension = 0.0728
        self.surface_threshold = 7.065
        self.force_net = np.array((0, 0))
    

    def update_density(self, others):
        pos = np.array([other.position for other in others])
        diff = pos - self.position

        self.density = self.mass * np.sum(self.kernel_poly_6(diff))
        self.pressure = self.gas_stiffness * (self.density - self.rest_density)


    def update_force(self, others)
        positions = np.array([other.position for other in others])
        pressures = np.array([other.pressure for other in others])
        densities = np.array([other.density for other in others])
        velocites = np.array([other.velocity for other in others])
        diff = positions - self.position

        force_pressure = -self.mass * self.density * np.sum((self.pressure / self.density ** 2) + (pressures / densities ** 2) * self.kernel_gradient_spiky(diff), axis=0)
        force_viscosity = self.viscosity * self.mass * np.sum((velocites - self.velocity) * self.kernel_laplacian_viscosity(diff) / densities, axis=0)
        
        colorfield_normal = self.mass * np.sum(self.kernel_gradient_poly_6(diff) / densities, axis=0)
        colorfield_magnitude = np.linalg.norm(colorfield_normal)
        if colorfield_magnitude >= self.surface_threshold:
            colorfield_laplacian = self.mass * np.sum(self.kernel.kernel_laplacian_poly_6(diff) / densities)
            force_surface = -self.surface_tension * colorfield_normal / colorfield_magnitude * colorfield_laplacian 
        else:
            force_surface = np.array((0, 0))

        force_gravity = np.array((0, self.density * GRAVITY_CONSTANCE))

        self.force_net = force_pressure + force_viscosity + force_surface + force_gravity
    

    def update_position(self, interval):
        acceleration = self.force_net / self.density
        position = self.velocity * interval + self.acceleration * (interval ** 2)
        velocity = (position - self.position) / interval
        self.position = position
        self.velocity = velocity


from collections import defaultdict
from itertools import product

class ParticelPool:
    def __init__(self, num_particles, factory):
        self.parts = [factory() for _ in range(num_particles)]
        self.h = self.parts[0].h
        self.grid = defaultdict(list)


    def update_grid(self):
        self.grid = defaultdict(list)
        for part in self.parts:
            grid_position = tuple((part.position / self.h).astype(np.int64).tolist())
            self.grid[grid_position].append(part)


    def update(self, interval):
        self.update_grid()

        for grid_position in self.grid:
            neighbors = []
            gx, gy = grid_position
            for dgx, dgy in product([-1, 0, 1], [-1, 0, 1]):
                neighbors.extend(self.grid.get((gx + dgx, gy + dgy), []))
            for part in self.grid[grid_position]:
                real_neighbors = neighbors[:]
                real_neighbors.remove(part)
                part.update_density(real_neighbors)
        
        for grid_position in self.grid:
            neighbors = []
            gx, gy = grid_position
            for dgx, dgy in product([-1, 0, 1], [-1, 0, 1]):
                neighbors.extend(self.grid.get((gx + dgx, gy + dgy), []))
            for part in self.grid[grid_position]:
                real_neighbors = neighbors[:]
                real_neighbors.remove(part)
                part.update_force(real_neighbors)
        
        for part in self.parts:
            part.update_position(interval)