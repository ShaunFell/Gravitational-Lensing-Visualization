import matplotlib.pyplot as plt
import matplotlib.image as img
import matplotlib.patches as ptch
from matplotlib.widgets import Slider
from matplotlib.widgets import Button
import astropy.constants as c
import astropy.units as u
import random
import numpy as np
plt.style.use('grayscale')



npoints = 5000
angle = [random.uniform(0,1)*np.pi*2 for i in np.arange(npoints)]
radius_ran = [random.uniform(0,1) for i in np.arange(npoints)]
x = lambda center_x, radius: [radius*radius_ran[i]*np.cos(angle[i]) + center_x for i in np.arange(npoints)]
y = lambda center_y, radius: [radius*radius_ran[i]*np.sin(angle[i]) + center_y for i in np.arange(npoints)]


# Main Calculations

D_S = 5000 #parsecs
D_L = 2000 #parsecs
D_LS = np.abs(D_S-D_L) #parsecs. distance from source to hole
G = c.G.value #gravitational constant
M = 10**11 #solar masses. Mass of lensing hole
C = c.c.value #m/s
R_E = np.sqrt((4*G*M*D_LS)/((C**2)*D_L*D_S))*D_L*3600 #Einstein Radius 
co_ords = (-R_E/2,-R_E/2) #co-ordinates for center of source object. This is the default value
radius = R_E/10 #Radius of source object


circle = {"x":np.asarray(x(co_ords[0],radius)), "y":np.asarray(y(co_ords[1],radius))} #Source image. Coord system is relative to lensing hole
r = np.sqrt(circle["x"]**2 + circle["y"]**2) # projected distance from each pixel in source image to lensing hole
beta = r/D_L/3600

theta1 = ((beta/2) + np.sqrt(((beta**2)/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S)))) #First lensed image angle
theta2 = ((beta/2) - np.sqrt(((beta**2)/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S)))) #Second lensed image angle
lensed_one_x, lensed_one_y = circle["x"]*theta1/beta, circle["y"]*theta1/beta #First lensed image
lensed_two_x, lensed_two_y = circle["x"]*theta2/beta, circle["y"]*theta2/beta #Second lensed image



#Setup plotting

max_x = np.max(R_E)*(1+0.2) #Bounds of plot
max_y = np.max(R_E)*(1+0.2)


#@Set plot parameters
fig, (ax, empty) = plt.subplots(2,1,gridspec_kw={'height_ratios':[12,1]}, figsize=(9,9))
empty.axis('off')
ax.set_xticks(np.arange(start = -R_E*1.2, stop = R_E*1.2, step = R_E/2))
ax.set_yticks(np.arange(start = -R_E*1.2, stop = R_E*1.2, step = R_E/2))
ax.tick_params(axis='x', rotation=30)
ax.set_xlabel(r"$\theta$ (mas)")
ax.set_ylabel(r"$\theta$ (mas)")
domain = np.arange(start=0, stop=2*np.pi, step=np.pi/24)
#ax.axes().set_aspect('equal', 'datalim')
ax.set_xlim(-max_x*(1+1.5), max_x*(1+1.5))
ax.set_ylim(-max_y*(1+1.5), max_y*(1+1.5))
ax.set_title(r"Gravitational lensing due to $10^{11}$ Solar mass object")



source, = ax.plot(circle["x"], circle["y"], label="source", linestyle='none', marker='.', color='#BFAC4D')
len1, = ax.plot(lensed_one_x, lensed_one_y, color='#E1CA57', linestyle='none', marker='.', label="lensed source")
len2, = ax.plot(lensed_two_x, lensed_two_y, linestyle='none', marker='.', color='#E1CA57')
ax.plot(R_E*np.sin(domain), R_E*np.cos(domain), linestyle='none', marker='.', markersize = 3, markerfacecolor='gray', label="Einstein Ring")


ax_xcoord = plt.axes([0.20,0.03,0.65,0.03])
ax_ycoord = plt.axes([0.20,0.09,0.65,0.03])
ax_loc1 = plt.axes([0.10, 0.13, 0.2,0.03])
ax_loc2 = plt.axes([0.10, 0.16, 0.2, 0.03])
ax_loc3 = plt.axes([0.10, 0.19, 0.2, 0.03])

#Init adjustable parameter sliders
s_xcoord = Slider(ax_xcoord, "X co-ordinate", -max_x, max_x, valinit = co_ords[0], valstep = co_ords[0]/100, valfmt='%0.4f')
s_ycoord = Slider(ax_ycoord, "Y co-ordinate", -max_y, max_y, valinit = co_ords[1], valstep = co_ords[1]/100, valfmt='%0.4f')
ax_loc1_button = Button(ax_loc1, "Center")
ax_loc2_button = Button(ax_loc2, "Einstein Radius")
ax_loc3_button = Button(ax_loc3, "Outside Einstein Radius")


def update(val):
    """
        Function updates source and lensed images based on new parameters provided by sliders.
    """
    x_coord = s_xcoord.val
    y_coord = s_ycoord.val
    circle = {"x":np.asarray(x(x_coord,radius)), "y":np.asarray(y(y_coord,radius))}
    r = np.sqrt(circle["x"]**2 + circle["y"]**2)
    beta = r/D_L/3600
    theta1 = ((beta/2) + np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
    theta2 = ((beta/2) - np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
    lensed_one_x = circle["x"]*theta1/beta
    lensed_one_y = circle["y"]*theta1/beta
    lensed_two_x = circle["x"]*theta2/beta
    lensed_two_y = circle["y"]*theta2/beta
    source.set_xdata(circle["x"])
    source.set_ydata(circle["y"])
    len1.set_xdata(lensed_one_x)
    len1.set_ydata(lensed_one_y)
    len2.set_xdata(lensed_two_x)
    len2.set_ydata(lensed_two_y)
	
class bclass(object):
    """
        Class handles button presses
    """
    def loc1(self, event):
        """
            Set source image location to align with lensing hole
        """
        x_coord = 0
        y_coord = 0
        circle = {"x":np.asarray(x(x_coord,radius)), "y":np.asarray(y(y_coord,radius))}
        r = np.sqrt(circle["x"]**2 + circle["y"]**2)
        beta = r/D_L/3600
        theta1 = ((beta/2) + np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
        theta2 = ((beta/2) - np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
        lensed_one_x = circle["x"]*theta1/beta
        lensed_one_y = circle["y"]*theta1/beta
        lensed_two_x = circle["x"]*theta2/beta
        lensed_two_y = circle["y"]*theta2/beta
        source.set_xdata(circle["x"])
        source.set_ydata(circle["y"])
        len1.set_xdata(lensed_one_x)
        len1.set_ydata(lensed_one_y)
        len2.set_xdata(lensed_two_x)
        len2.set_ydata(lensed_two_y)
    def loc2(self, event):
        """
            Set source image location to lie on Einstein radius
        """
        x_coord = co_ords[0]
        y_coord = co_ords[1]
        circle = {"x":np.asarray(x(x_coord,radius)), "y":np.asarray(y(y_coord,radius))}
        r = np.sqrt(circle["x"]**2 + circle["y"]**2)
        beta = r/D_L/3600
        theta1 = ((beta/2) + np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
        theta2 = ((beta/2) - np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
        lensed_one_x = circle["x"]*theta1/beta
        lensed_one_y = circle["y"]*theta1/beta
        lensed_two_x = circle["x"]*theta2/beta
        lensed_two_y = circle["y"]*theta2/beta
        source.set_xdata(circle["x"])
        source.set_ydata(circle["y"])
        len1.set_xdata(lensed_one_x)
        len1.set_ydata(lensed_one_y)
        len2.set_xdata(lensed_two_x)
        len2.set_ydata(lensed_two_y)
    def loc3(self, event):
        """
            Set source image location outside Einstein radius
        """
        x_coord = R_E
        y_coord = R_E
        circle = {"x":np.asarray(x(x_coord,radius)), "y":np.asarray(y(y_coord,radius))}
        r = np.sqrt(circle["x"]**2 + circle["y"]**2)
        beta = r/D_L/3600
        theta1 = ((beta/2) + np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
        theta2 = ((beta/2) - np.sqrt((beta**2/4) + ((4*G*M*D_LS)/((C**2)*D_L*D_S))))
        lensed_one_x = circle["x"]*theta1/beta
        lensed_one_y = circle["y"]*theta1/beta
        lensed_two_x = circle["x"]*theta2/beta
        lensed_two_y = circle["y"]*theta2/beta
        source.set_xdata(circle["x"])
        source.set_ydata(circle["y"])
        len1.set_xdata(lensed_one_x)
        len1.set_ydata(lensed_one_y)
        len2.set_xdata(lensed_two_x)
        len2.set_ydata(lensed_two_y)

container = bclass()
ax_loc1_button.on_clicked(container.loc1)
ax_loc2_button.on_clicked(container.loc2)
ax_loc3_button.on_clicked(container.loc3)
s_xcoord.on_changed(update)
s_ycoord.on_changed(update)

ax.legend()
#plt.tight_layout()
plt.show()






