#    hairSEM aims to quantify hair damage using a SEM image
#    Copyright (C) 2024 dolphin2410
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import math
from settings import X_SIZE, Y_SIZE


class LinearGraph:
    def __init__(self, p1, p2):
        if p1[0] == p2[0] and p1[1] == p2[1]:
            raise ValueError("The given two points are identical")
        
        self.p1 = p1
        self.p2 = p2

        if p1[0] == p2[0]:
            self.gradient = math.inf
        else:
            self.gradient = (p1[1] - p2[1]) / (p1[0] - p2[0])

    def linear_graph_coeffs(self):
        """this function takes two points as an input and returns a, b values of the equation ax + by = 1"""
        
        x1, y1 = self.p1
        x2, y2 = self.p2

        a = (y2 - y1) / (x1 * y2 - x2 * y1)
        b = (x2 - x1) / (x2 * y1 - x1 * y2)
        
        return (a, b)
    
    def boundary_intercepts(self):
        """Returns the intercepts of the LinearGraph within the given boundary as points"""

        a, b = self.linear_graph_coeffs()
        intercepts = []

        if b <= 0: 
            intercepts.append((int(1/a), 0))
        else:
            intercepts.append((0, int(1/b)))
        
        if b == 0 or 1 - a * X_SIZE > Y_SIZE * b:
            intercepts.append((int((1 - b * Y_SIZE) / a), Y_SIZE))
        else:
            intercepts.append((X_SIZE, int((1 - a * X_SIZE) / b)))

        return list(set(intercepts))

    def perpendicular_gradient(self):
        if self.gradient == 0:
            return 1000 # i wish this would be enough?
        else:
            return - 1.0 / self.gradient
        
        
        
