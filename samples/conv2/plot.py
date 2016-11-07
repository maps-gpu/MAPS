#!/usr/bin/env python

# Plots the result of conv2 as relative speedup bar graph

import sys

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np

if len(sys.argv) != 2:
    print('USAGE: plot.py <output.log>')
    exit(0)
logfile = sys.argv[1]

width = 0.35 # bar width
stride =  2  # spacing between each bar

yres = {'Naive': [], 'Optimized': [], 'MAPS': []}

# Parse results to obtain bar contents
with open(logfile, 'r') as f:
    for line in f:
        parse = line.split()
        # If this is a result
        if (len(parse) > 0 and parse[-1] == 'ms'):
            yres[parse[0]].append(parse[-2])
    
yunopt = np.array([float(y) for y in yres['Naive']])
yopt   = np.array([float(y) for y in yres['Optimized']])
ymaps  = np.array([float(y) for y in yres['MAPS']])

# Relative speedup computation
yopt   = yunopt / yopt
ymaps  = yunopt / ymaps
yunopt = yunopt / yunopt

# Set titles and x positions for bars
titles = ['1x1', '3x3', '5x5', '7x7', '9x9', '11x11', '13x13']
xunopt = [i*stride for i in range(len(titles))]
xopt   = [x+width for x in xunopt]
xmaps  = [x+width for x in xopt]
xtics  = [x+(width/2) for x in xopt]


# Plot
fig, ax = plt.subplots()
rects_unopt = ax.bar(xunopt, yunopt, width, color=(31/255.0, 119/255.0, 180/255.0))
rects_opt   = ax.bar(xopt,   yopt,   width, color=(255/255.0, 127/255.0, 14/255.0))
rects_maps  = ax.bar(xmaps,  ymaps,  width, color=(44/255.0, 160/255.0, 44/255.0))

# Set graph labels and legend
ax.set_ylabel('Relative Speedup')
ax.set_title('2D Convolution Performance')
ax.set_xticks(xtics)
ax.set_xticklabels(titles)
ax.legend((rects_unopt[0], rects_opt[0], rects_maps[0]), ('Naive', 'Optimized', 'MAPS'), loc=0)

# Save resulting plot
plt.savefig('performance.png')
