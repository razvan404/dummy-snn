import numpy as np
import torch
from scipy import stats as sp_stats


class NeuronTracker:
    """Collects per-neuron training trajectory and computes derived features.

    Hooks into the training loop via on_batch_end and on_epoch_end callbacks.
    After training, call compute_trajectory_features() to get derived metrics.
    """

    def __init__(
        self,
        n_neurons: int,
        n_epochs: int,
        n_classes: int = 10,
        weight_snapshot_interval: int = 10,
    ):
        self.n_neurons = n_neurons
        self.n_epochs = n_epochs
        self.n_classes = n_classes
        self.weight_snapshot_interval = weight_snapshot_interval

        # Per-epoch trajectory arrays
        self.threshold_history = np.zeros((n_epochs, n_neurons))
        self.win_counts = np.zeros((n_epochs, n_neurons))
        self.win_counts_per_class = np.zeros((n_epochs, n_neurons, n_classes))
        self.cumulative_stdp_mag = np.zeros((n_epochs, n_neurons))

        # Weight snapshots at checkpoints
        self.weight_snapshots: list[np.ndarray] = []
        self.weight_snapshot_epochs: list[int] = []

        # Internal state
        self._current_epoch = 0

    def on_epoch_start(self, epoch: int, layer: torch.nn.Module) -> None:
        """Record threshold snapshot at start of epoch."""
        self._current_epoch = epoch
        self.threshold_history[epoch] = layer.thresholds.detach().cpu().numpy()

    def on_batch(
        self,
        learner,
        label: int | None,
        dw: float,
    ) -> None:
        """Called after each training batch. Records winner and STDP magnitude.

        Args:
            learner: The Learner object (has .neurons_to_learn attribute).
            label: Class label of the current sample.
            dw: Mean absolute weight change from STDP.
        """
        epoch = self._current_epoch
        neurons = learner.neurons_to_learn

        if len(neurons) > 0:
            for idx in neurons:
                neuron_idx = idx.item()
                self.win_counts[epoch, neuron_idx] += 1
                if label is not None and 0 <= label < self.n_classes:
                    self.win_counts_per_class[epoch, neuron_idx, label] += 1
                self.cumulative_stdp_mag[epoch, neuron_idx] += dw

    def on_epoch_end(self, epoch: int, layer: torch.nn.Module) -> None:
        """Snapshot weights at checkpoint intervals."""
        if (
            epoch + 1
        ) % self.weight_snapshot_interval == 0 or epoch == self.n_epochs - 1:
            weights = layer.weights.detach().cpu().numpy().copy()
            self.weight_snapshots.append(weights)
            self.weight_snapshot_epochs.append(epoch)

    def compute_trajectory_features(self) -> dict[str, np.ndarray]:
        """Compute all trajectory-derived features after training.

        Returns:
            Dict mapping feature_name -> array of shape (n_neurons,).
        """
        features = {}

        # Total STDP magnitude over all epochs
        features["cumulative_stdp_magnitude"] = self.cumulative_stdp_mag.sum(axis=0)

        # Threshold integral: sum of thresholds over all epochs
        features["threshold_integral"] = self.threshold_history.sum(axis=0)

        # Threshold variance in last 20% of epochs
        late_start = max(1, int(self.n_epochs * 0.8))
        late_thresholds = self.threshold_history[late_start:]
        features["threshold_variance_late"] = (
            late_thresholds.var(axis=0)
            if late_thresholds.shape[0] > 1
            else np.zeros(self.n_neurons)
        )

        # Win entropy: Shannon entropy of per-class win distribution
        total_wins_per_class = self.win_counts_per_class.sum(axis=0)  # (N, C)
        features["win_entropy"] = self._compute_entropy(total_wins_per_class)

        # Weight velocity in last 10% of training
        features["weight_velocity_late"] = self._compute_weight_velocity_late()

        # Threshold trend slope: linear regression over last 50% of training
        features["threshold_trend_slope"] = self._compute_threshold_slope()

        # Convergence epoch: first epoch where weight cosine similarity > 0.99
        # for 3 consecutive snapshots
        features["convergence_epoch"] = self._compute_convergence_epoch()

        # Total win count
        features["total_win_count"] = self.win_counts.sum(axis=0)

        return features

    def _compute_entropy(self, counts_per_class: np.ndarray) -> np.ndarray:
        """Shannon entropy of per-class win distribution for each neuron."""
        entropy = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            total = counts_per_class[i].sum()
            if total <= 0:
                continue
            probs = counts_per_class[i] / total
            probs = probs[probs > 0]
            entropy[i] = -np.sum(probs * np.log(probs))
        return entropy

    def _compute_weight_velocity_late(self) -> np.ndarray:
        """Mean weight change magnitude in last 10% of training epochs."""
        if len(self.weight_snapshots) < 2:
            return np.zeros(self.n_neurons)

        # Use last ~10% of snapshots
        n_late = max(1, len(self.weight_snapshots) // 10)
        late_snapshots = self.weight_snapshots[-n_late - 1 :]

        velocities = []
        for i in range(1, len(late_snapshots)):
            diff = np.abs(late_snapshots[i] - late_snapshots[i - 1])
            velocities.append(diff.mean(axis=1))  # (n_neurons,)

        if not velocities:
            return np.zeros(self.n_neurons)
        return np.mean(velocities, axis=0)

    def _compute_threshold_slope(self) -> np.ndarray:
        """Linear regression slope of threshold over last 50% of epochs."""
        half_start = max(1, self.n_epochs // 2)
        late_thresholds = self.threshold_history[half_start:]
        n_points = late_thresholds.shape[0]

        if n_points < 2:
            return np.zeros(self.n_neurons)

        x = np.arange(n_points, dtype=np.float64)
        slopes = np.zeros(self.n_neurons)
        for i in range(self.n_neurons):
            slope, _, _, _, _ = sp_stats.linregress(x, late_thresholds[:, i])
            slopes[i] = slope
        return slopes

    def _compute_convergence_epoch(self) -> np.ndarray:
        """First epoch where weight cosine similarity > 0.99 for 3 consecutive snapshots."""
        convergence = np.full(self.n_neurons, float(self.n_epochs))

        if len(self.weight_snapshots) < 4:
            return convergence

        consecutive = np.zeros(self.n_neurons, dtype=int)
        for i in range(1, len(self.weight_snapshots)):
            prev = self.weight_snapshots[i - 1]
            curr = self.weight_snapshots[i]

            # Per-neuron cosine similarity
            dot = np.sum(prev * curr, axis=1)
            norm_prev = np.linalg.norm(prev, axis=1)
            norm_curr = np.linalg.norm(curr, axis=1)
            denom = norm_prev * norm_curr
            denom = np.where(denom < 1e-10, 1e-10, denom)
            cos_sim = dot / denom

            converged = cos_sim > 0.99
            consecutive = np.where(converged, consecutive + 1, 0)

            first_converged = (consecutive >= 3) & (convergence == self.n_epochs)
            convergence[first_converged] = self.weight_snapshot_epochs[i]

        return convergence
