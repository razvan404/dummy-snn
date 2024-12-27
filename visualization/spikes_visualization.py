import numpy as np
from matplotlib import pyplot as plt

from spiking import Spike

from .consts import FIG_SIZE


class SpikesVisualization:
    @classmethod
    def plot_spikes(cls, spikes: list[Spike], title: str | None = None):
        plt.figure(figsize=FIG_SIZE)
        for spike in spikes:
            plt.axvline(x=spike.time, color="blue", alpha=0.6, linewidth=0.8)
        if title:
            plt.title(title)
        plt.xlabel("Time")
        plt.ylabel("Amplitude")
        plt.yticks([])
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.show()

    @classmethod
    def plot_pre_post_spikes(
        cls,
        pre_spike_times: list[float],
        post_spike_times: list[float],
        title: str | None = None,
        unique_colors: bool = False,
    ):
        plt.figure(figsize=FIG_SIZE)

        pre_y = np.zeros_like(pre_spike_times)

        plt.scatter(pre_spike_times, pre_y, color="blue", label="Pre-synaptic Spikes")
        if not unique_colors:
            post_y = np.ones_like(post_spike_times)
            plt.scatter(
                post_spike_times, post_y, color="red", label="Post-synaptic Spikes"
            )
        else:
            cmap = plt.cm.get_cmap("viridis", len(post_spike_times))
            for idx, time in enumerate(post_spike_times):
                plt.scatter(
                    time, 1, color=cmap(idx), label=f"Post-synaptic Spike {idx}"
                )

        plt.yticks([0, 1], ["Pre-synaptic", "Post-synaptic"])
        plt.xlabel("Time (ms)")
        plt.title("Spike Times: Pre-synaptic vs Post-synaptic")
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        if title:
            plt.title(title)
        plt.show()

    @classmethod
    def plot_membrane_potentials(
        cls, potentials: list[float], times: list[float], title: str | None = None
    ):
        plt.plot(
            times,
            potentials,
            label="Membrane Potential",
            color="blue",
            marker="o",
            linestyle="-",
            markersize=5,
        )

        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (V)")
        if title:
            plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.show()

    @classmethod
    def plot_multiple_membrane_potentials(
        cls,
        potentials_lists: list[list[float]],
        times: list[float],
        title: str | None = None,
    ):
        potentials_lists = list(zip(*potentials_lists))
        cmap = plt.cm.get_cmap("viridis", len(potentials_lists))
        for idx, potentials in enumerate(potentials_lists):
            plt.plot(
                times,
                potentials,
                label=f"Membrane Potential {idx}",
                color=cmap(idx),
                marker="o",
                linestyle="-",
                markersize=5,
            )

        plt.xlabel("Time (ms)")
        plt.ylabel("Membrane Potential (V)")
        if title:
            plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()

        plt.show()
