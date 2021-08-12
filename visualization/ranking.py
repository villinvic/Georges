import matplotlib.image as image
import matplotlib.pyplot as plt
import numpy as np

plt.rc('font', family='serif')

def visualize(population, path):

    elos = []
    char_paths = []
    names = []

    for individual in population:
        elos.append(individual.elo())
        char_paths.append(individual.genotype['type'].get().ico)
        names.append(individual.name.get())

    # constant
    fig, ax = plt.subplots()
    fig.set_size_inches(18., 8.5)
    ax.set_ylabel('Elo')
    ax.axis([-1, len(elos), np.min(elos)-100, np.max(elos)+100])
    ax.set_title('Population Elo')

    ax.axes.get_xaxis().set_visible(False)

    points = ax.scatter(range(len(elos)),elos)

    #coords = points.get_offsets()

    #coords[:, 0] /= len(elos) * 0.3665
    #coords[:, 1] = 1.85 * (coords[:, 1]-np.min(elos) ) / ( 1e-5 + np.max(elos)-np.min(elos))

    xy_pixels = ax.transData.transform(points.get_offsets())
    xpix, ypix = xy_pixels.T

    d = (np.max(elos) - np.min(elos) + 200)*0.01


    font = {
            'size'  : 8,
            }

    for xp, yp in zip(xpix, ypix):
        print('{x:0.2f}\t{y:0.2f}'.format(x=xp, y=yp))

    for i, (ico_path, name, elo) in enumerate(zip(char_paths, names, elos)):
        im = image.imread('characters/' + ico_path)

        fig.figimage(im, xpix[i]-12, ypix[i]-12)

        ax.text(i-len(name)*0.076, elo-d*4, name, fontdict=font)

    fig.savefig(path+'Elo_visualization.pdf')














