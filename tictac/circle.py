#import 

class Circle :
    def __init__(self,position,  radius,dx,dy) :
        self.position = position
        self.radius = radius
        self.dx = dx
        self.dy = dy
    
    def move(self,speed,dt) :
        self.position.set_x(speed*dt*self.dx)
        self.position.set_y(self.dy*speed*dt)
    def get_x(self) :
        return self.position.get_x()
    def get_y(self) :
        return self.position.get_y()
    def get_radius(self) :
        return self.radius
    def set_dx(self,dx) :
        self.dx = dx
    def set_dy(self,dy) :
        self.dy = dy
    def get_dx(self) :
        return self.dx
    def get_dy(self) :
        return self.dy


class Coordinates:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def get_x(self):
        return self.x
    def get_y(self):
        return self.y
    def set_x(self,dx):
        self.x += dx
    def set_y(self,dy):
        self.y += dy