from settings import X_SIZE, Y_SIZE


class LinearGraph:
    def __init__(self, p1, p2):
        if p1[0] == p2[0] and p1[1] == p2[1]:
            raise ValueError("The given two points are identical")
        
        self.p1 = p1
        self.p2 = p2

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

        
        
        
