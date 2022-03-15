#GIF creation
import imageio

names = [i for i in range(1, 250, 5)] + [i for i in range(250, 275, 1)] + [i for i in range(275, 501, 5)] + [503 for i in range(10)]
filenames = [f'Figures/{names[i]}.png' for i in range(len(names))]

images = []
for filename in filenames:
    images.append(imageio.imread(filename))
imageio.mimsave('Figures/movie4.gif', images)
