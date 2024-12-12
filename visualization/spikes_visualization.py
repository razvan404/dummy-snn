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
    ):
        plt.figure(figsize=FIG_SIZE)

        pre_y = np.zeros_like(pre_spike_times)
        post_y = np.ones_like(post_spike_times)

        plt.scatter(pre_spike_times, pre_y, color="blue", label="Pre-synaptic Spikes")
        plt.scatter(post_spike_times, post_y, color="red", label="Post-synaptic Spikes")

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
