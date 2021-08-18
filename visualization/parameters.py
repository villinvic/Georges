import numpy as np
import matplotlib.pyplot as plt
from config.loader import Default
from population.population import Individual
from game.enums import PlayerType

class IndividualVisualizer(Default):

    def __init__(self):
        super(IndividualVisualizer, self).__init__()


    def observe(self, individual:Individual):

        if individual.type == PlayerType.CPU:
            return

        plt.rc('font', family='serif')

        fig, ax = plt.subplots()

        all_params = dict(individual.genotype['learning']._variables)
        all_params.update(individual.genotype['experience']._variables)

        gene_values =  [(v.get() - v.domain[0])/(v.domain[1]-v.domain[0]) for v in all_params.values()]

        ys = np.arange(len(all_params))

        colors = ['b'] * len(individual.genotype['experience']._variables) \
                 +['r'] * len(individual.genotype['learning']._variables)

        bars = ax.barh(ys, gene_values, color=colors, log=True)

        ax.set_yticks(ys)
        #ax.set_yticklabels(gene_names)
        #ax.invert_yaxis()
        ax.set_xlim([1e-5,1])
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

        ax.set_title(individual.name.get().replace('$', '\$').replace('_', '\_') + '\'s Genotype')

        labels = [gene.name.replace('_', '\_')+'=%.5f' % gene.get() for gene in all_params.values()]
        for i, label in enumerate(reversed(labels)):
            ax.text(0, 1000+i, label)

        fig.savefig(self.output_path)
        plt.close(fig)