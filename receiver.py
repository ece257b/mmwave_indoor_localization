#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
from scipy.optimize import minimize
import seaborn as sns


# -------------------------------------------------------------------------
# CFO synchronization and path extraction
# -------------------------------------------------------------------------
def cfo_time_sync_and_extract_paths(tx_baseband, rx_baseband, pilot_len, Fs):
    """
    Estimate time offset, CFO, correct them, then extract path parameters.
    """
    pilot = tx_baseband[:pilot_len]
    corr = np.correlate(rx_baseband, pilot.conjugate(), mode='full')
    peak_idx = np.argmax(np.abs(corr))
    time_offset = peak_idx - (pilot_len - 1)
    if time_offset < 0:
        time_offset = 0

    rx_aligned = rx_baseband[time_offset: time_offset + len(tx_baseband)]
    if len(rx_aligned) < len(tx_baseband):
        rx_aligned = np.pad(rx_aligned, (0, len(tx_baseband) - len(rx_aligned)), 'constant')

    half = pilot_len // 2
    r1 = rx_aligned[:half]
    r2 = rx_aligned[half:pilot_len]
    cross = np.vdot(r1, r2)
    phase_diff = np.angle(cross)
    cfo_est = phase_diff / (2 * np.pi * (half / Fs))

    n = np.arange(len(rx_aligned))
    cfo_corr_phasor = np.exp(-1j * 2 * np.pi * cfo_est * n / Fs)
    rx_corrected = rx_aligned * cfo_corr_phasor

    full_corr = np.correlate(rx_corrected, pilot.conjugate(), mode='full')
    corr_abs = np.abs(full_corr)
    n_peaks = 3
    peak_indices = np.argpartition(corr_abs, -n_peaks)[-n_peaks:]
    peak_indices = peak_indices[np.argsort(-corr_abs[peak_indices])]

    path_delays_est = []
    path_powers_est = []
    for idx in peak_indices:
        delay_samps = idx - (pilot_len - 1)
        if delay_samps >= 0:
            path_delays_est.append(delay_samps / Fs)
            path_powers_est.append(corr_abs[idx])

    combined = sorted(zip(path_delays_est, path_powers_est), key=lambda x: x[0])
    if len(combined) > 0:
        path_delays_est, path_powers_est = map(list, zip(*combined))
    else:
        path_delays_est, path_powers_est = [], []

    return rx_corrected, cfo_est, time_offset, path_delays_est, path_powers_est

# -------------------------------------------------------------------------
# Extract path parameters for all users
# -------------------------------------------------------------------------
def process_channel_data(ap_positions, user_positions, ap_selected, tx_baseband, rx_basebands,
                         pilot_len=64, Fs=1e6, n_paths=3):
    estimations = []
    for i, (user_pos, rx_baseband) in enumerate(zip(user_positions, rx_basebands)):
        ap_idx = ap_selected[i]
        ap_pos = ap_positions[ap_idx]

        dir_vector = user_pos - ap_pos
        dx, dy = dir_vector[0], dir_vector[1]
        angle_los = np.degrees(np.arctan2(dy, dx))
        estimated_angles = [angle_los] * n_paths

        rx_corrected, cfo_est, time_offset, path_delays_est, path_powers_est = cfo_time_sync_and_extract_paths(
            tx_baseband, rx_baseband, pilot_len, Fs
        )

        if len(path_delays_est) == 0:
            path_delays_est = [0.0]
            path_powers_est = [1.0]
            estimated_angles = [0.0]

        p_sort_idx = np.argsort(path_powers_est)[::-1]
        path_delays_est = np.array(path_delays_est)[p_sort_idx][:n_paths].tolist()
        path_powers_est = np.array(path_powers_est)[p_sort_idx][:n_paths].tolist()
        estimated_angles = estimated_angles[:n_paths]

        DoA = []
        DoD = []
        DDoF = []
        Power = []
        Alpha_r = []
        Alpha_i = []

        dist = np.linalg.norm(dir_vector)
        doa_3d = dir_vector / dist if dist > 1e-9 else np.array([1,0,0])
        dod_3d = -doa_3d
        DDoF.append(0.0)
        p_lin_max = path_powers_est[0]
        p_db_max = 10 * np.log10(p_lin_max + 1e-12)
        Power.append(p_db_max)
        phase_0 = np.random.uniform(0, 2*np.pi)
        amp_0 = 10 ** (p_db_max/20)
        Alpha_r.append(amp_0 * np.cos(phase_0))
        Alpha_i.append(amp_0 * np.sin(phase_0))
        DoA.append(doa_3d.tolist())
        DoD.append(dod_3d.tolist())

        for p_idx in range(1, len(path_delays_est)):
            delay_rel = path_delays_est[p_idx] - path_delays_est[0]
            DDoF.append(delay_rel)
            p_lin = path_powers_est[p_idx]
            p_db = 10 * np.log10(p_lin + 1e-12)
            Power.append(p_db)
            amp = 10 ** (p_db/20)
            phase = np.random.uniform(0, 2*np.pi)
            Alpha_r.append(amp * np.cos(phase))
            Alpha_i.append(amp * np.sin(phase))
            angle_rad = np.radians(estimated_angles[p_idx])
            dx = np.cos(angle_rad)
            dy = np.sin(angle_rad)
            dz = doa_3d[2] + np.random.uniform(-0.1, 0.1)
            refl_dir = np.array([dx, dy, dz])
            refl_dir /= np.linalg.norm(refl_dir)
            DoA.append(refl_dir.tolist())
            DoD.append((-refl_dir).tolist())

        estimation = {
            'DoA': DoA,
            'DoD': DoD,
            'DDoF': DDoF,
            'Power': Power,
            'Alpha_r': Alpha_r,
            'Alpha_i': Alpha_i,
            'user_position': user_pos.tolist()
        }
        estimations.append(estimation)

    return estimations

def classify_paths(DoA, DDoF):
    DoA = np.array(DoA)
    DDoF = np.array(DDoF)
    classifications = ["LoS"]
    for i in range(1, len(DoA)):
        if abs(DoA[i][2] - DoA[0][2]) < 0.15:
            classifications.append("vertical")
        elif np.linalg.norm(DoA[i][:2] - DoA[0][:2]) < 0.15:
            classifications.append("horizontal")
        else:
            classifications.append("other")
    return classifications

def initial_localization_with_classification(ap_pos, DoA, DDoF, weights=None, max_err=0.2):
    c = 3e8
    DoA = np.array(DoA)
    DDoF = np.array(DDoF)
    if weights is None:
        weights = np.ones(len(DoA))
    def cost_function(tau_0):
        total_cost = 0
        for l in range(1, len(DoA)):
            c_vertical = abs(c*(DDoF[l] + tau_0)*DoA[l][2] - c*(DDoF[0] + tau_0)*DoA[0][2])
            horizontal_diff = c*(DDoF[l] + tau_0)*DoA[l][:2] - c*(DDoF[0] + tau_0)*DoA[0][:2]
            c_horizontal = np.linalg.norm(horizontal_diff)
            c_other = max_err
            path_cost = min(c_vertical, c_horizontal, c_other)
            total_cost += weights[l] * path_cost
        return total_cost
    result = minimize(cost_function, 0, method='BFGS')
    tau_0 = result.x[0]
    user_pos = ap_pos + c*(DDoF[0] + tau_0)*DoA[0]
    return user_pos

def classic_localization(ap_pos, DoA, DoD, DDoF, weights=None):
    c = 3e8
    DoA = np.array(DoA)
    DoD = np.array(DoD)
    DDoF = np.array(DDoF)
    if weights is None:
        weights = np.ones(len(DoA))
    A = []
    b = []
    doa = DoA[0]
    A.append(weights[0] * (np.eye(3) - np.outer(doa, doa)))
    b.append(weights[0] * np.dot(np.eye(3) - np.outer(doa, doa), ap_pos))
    for i in range(1, len(DoA)):
        doa = DoA[i]
        dod = DoD[i]
        delay = DDoF[i]
        normal = doa + dod
        normal_norm = np.linalg.norm(normal)
        if normal_norm > 1e-9:
            normal = normal / normal_norm
            A.append(weights[i] * (np.eye(3) - np.outer(doa, doa)))
            b.append(weights[i] * np.dot(np.eye(3) - np.outer(doa, doa), ap_pos + c*delay*doa))
            A.append(weights[i] * (np.eye(3) - np.outer(dod, dod)))
            b.append(weights[i] * np.dot(np.eye(3) - np.outer(dod, dod), ap_pos))
    A = np.vstack(A)
    b = np.concatenate(b)
    try:
        position, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        if np.any(np.isnan(position)) or np.linalg.norm(position - ap_pos) > 30:
            return None
        return position
    except:
        return None

def compute_virtual_anchor(ap_pos, user_pos, DoA, DDoF, path_idx):
    c = 3e8
    doa = np.array(DoA[path_idx])
    tau = DDoF[path_idx]
    dist_ap_user = np.linalg.norm(user_pos - ap_pos)
    virtual_user_pos = ap_pos + (tau + dist_ap_user) * doa
    normal = (user_pos - virtual_user_pos)
    norm_len = np.linalg.norm(normal)
    if norm_len < 1e-9:
        normal = np.array([0,0,1])
    else:
        normal /= norm_len
    reflection_dir = -doa + 2*normal*np.dot(normal, doa)
    virtual_anchor = user_pos + (tau + dist_ap_user)*(-reflection_dir)
    return virtual_anchor

def find_virtual_anchors(ap_positions, estimations, initial_positions, ap_selected):
    from sklearn.mixture import GaussianMixture
    total_anchors = [[] for _ in range(len(ap_positions))]
    total_weights = [[] for _ in range(len(ap_positions))]
    for est, init_pos, ap_idx in zip(estimations, initial_positions, ap_selected):
        DoA = np.array(est['DoA'])
        DDoF = np.array(est['DDoF'])
        Power = np.array(est['Power'])
        ap_pos = ap_positions[ap_idx]
        if np.linalg.norm(init_pos - np.array(est['user_position'])) > 2.0:
            continue
        for i in range(1, len(DoA)):
            try:
                va = compute_virtual_anchor(ap_pos, init_pos, DoA, DDoF, i)
                w = 10 ** ((Power[i] + 60)/20)
                total_anchors[ap_idx].append(va)
                total_weights[ap_idx].append(w)
            except:
                continue

    selected_anchors = []
    for ap_idx, (anchors, weights) in enumerate(zip(total_anchors, total_weights)):
        if len(anchors) < 5:
            selected_anchors.append(np.array([]))
            continue
        anchors = np.array(anchors)
        weights = np.array(weights)
        if np.sum(weights) > 0:
            weights /= np.sum(weights)
        n_components = min(5, len(anchors)//10)
        if n_components < 1:
            n_components = 1
        gmm = GaussianMixture(n_components=n_components, covariance_type='full', random_state=42)
        gmm.fit(anchors)
        va_centers = gmm.means_
        dist_to_ap = np.linalg.norm(va_centers - ap_positions[ap_idx], axis=1)
        va_centers = va_centers[dist_to_ap > 1.0]
        selected_anchors.append(va_centers)

    final_anchors = []
    for ap_idx, va in enumerate(selected_anchors):
        if len(va) > 0:
            all_anchors = np.vstack([ap_positions[ap_idx].reshape(1,3), va])
        else:
            all_anchors = ap_positions[ap_idx].reshape(1,3)
        final_anchors.append(all_anchors)
    return final_anchors

def virtual_anchor_localization(anchors, DoA, DDoF, weights=None, ang_err=3*np.pi/180, dist_err=0.2):
    from scipy.optimize import minimize
    DoA = np.array(DoA)
    DDoF = np.array(DDoF)
    if weights is None:
        weights = np.ones(len(anchors))
    if len(DoA) < 2 or len(anchors) < 2:
        return anchors[0] + DDoF[0]*DoA[0]
    best_pos = None
    best_score = float('-inf')
    max_paths = min(3, len(DoA))
    for n_paths in range(2, max_paths+1):
        path_indices = np.arange(min(n_paths, len(DoA)))
        anchor_indices = np.arange(min(n_paths, len(anchors)))
        pairs = []
        for path_idx in path_indices:
            for anchor_idx in anchor_indices:
                pairs.append((path_idx, anchor_idx))

        def cost_function(tau_0, pairs):
            tau_0 = tau_0[0]
            positions = []
            for path_idx, anchor_idx in pairs:
                doa = DoA[path_idx]
                delay = DDoF[path_idx]
                anchor = anchors[anchor_idx]
                if anchor_idx > 0:
                    ap_pos = anchors[0]
                    normal = anchor - ap_pos
                    norm_len = np.linalg.norm(normal)
                    if norm_len < 1e-9:
                        normal = np.array([0,0,1])
                    else:
                        normal /= norm_len
                    doa_reflected = doa - 2*normal*np.dot(normal, doa)
                    positions.append(anchor + (delay + tau_0)*doa_reflected)
                else:
                    positions.append(anchor + (delay + tau_0)*doa)
            mean_pos = np.mean(positions, axis=0)
            error = sum(np.linalg.norm(pos - mean_pos)**2 for pos in positions)
            return error

        result = minimize(cost_function, [0], args=(pairs,), method='BFGS')
        tau_0 = result.x[0]
        positions = []
        for path_idx, anchor_idx in pairs:
            doa = DoA[path_idx]
            delay = DDoF[path_idx]
            anchor = anchors[anchor_idx]
            if anchor_idx > 0:
                ap_pos = anchors[0]
                normal = anchor - ap_pos
                norm_len = np.linalg.norm(normal)
                if norm_len < 1e-9:
                    normal = np.array([0,0,1])
                else:
                    normal /= norm_len
                doa_reflected = doa - 2*normal*np.dot(normal, doa)
                positions.append(anchor + (delay + tau_0)*doa_reflected)
            else:
                positions.append(anchor + (delay + tau_0)*doa)
        final_pos = np.mean(positions, axis=0)
        variance = np.mean([np.linalg.norm(pos - final_pos) for pos in positions])
        score = -variance
        if score > best_score:
            best_score = score
            best_pos = final_pos

    if best_pos is None:
        best_pos = anchors[0] + DDoF[0]*DoA[0]
    return best_pos

# -------------------------------------------------------------------------
# Simple Neural Network Localizer
# -------------------------------------------------------------------------
class SimpleNNLocalizer:
    def __init__(self):
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(25,)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(3)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        return model

    def prepare_features(self, estimations):
        X = []
        y = []
        for est in estimations:
            doa = np.array(est['DoA'])
            ddof = np.array(est['DDoF'])
            power = np.array(est['Power'])
            features = []
            for i in range(min(5, len(doa))):
                features.extend(doa[i])
                features.append(ddof[i])
                features.append(power[i])
            while len(features) < 25:
                features.append(0)
            X.append(features)
            y.append(est['user_position'])
        return np.array(X), np.array(y)

    def train(self, X, y, epochs=100, batch_size=32, validation_split=0.2):
        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', patience=20, restore_best_weights=True
            )],
            verbose=1
        )
        return history

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y_true):
        y_pred = self.predict(X)
        position_error = np.sqrt(np.sum((y_pred - y_true)**2, axis=1))
        metrics = {
            'mean_position_error': np.mean(position_error),
            'median_position_error': np.median(position_error),
            'position_error_90th': np.percentile(position_error, 90),
            'position_errors': position_error
        }
        return metrics

    def plot_error_cdf(self, errors, label=None, color=None, linestyle=None):
        errors_cm = errors * 100
        sorted_errors = np.sort(errors_cm)
        cdf = np.arange(1, len(sorted_errors)+1) / len(sorted_errors) * 100
        plt.figure("Localization error")
        plt.plot(sorted_errors, cdf, label=label, color=color, linestyle=linestyle, linewidth=2)
        plt.xlabel("Localization error (cm)")
        plt.ylabel("Cumulative probability (%)")
        plt.grid(alpha=0.75, linestyle='--')
        plt.xlim([0, 400])

def main():



    ap_positions = []
    ap_positions.append([0, 4, 2.4])  # Left wall
    ap_positions.append([10, 4, 2.4])  # Right wall
    ap_positions = np.array(ap_positions[:2])

    try:
        interleaved_tx = np.fromfile("tx_signal.dat", dtype=np.float32)
        tx_baseband = interleaved_tx[0::2] + 1j * interleaved_tx[1::2]
        print("Processing: Loaded TX signal from 'tx_signal.dat'.")
    except Exception as e:
        print("Error loading 'tx_signal.dat':", e)
        return

    try:
        meta_data = np.load("user_positions.npz", allow_pickle=True)
        user_positions = meta_data['user_positions']
        ap_selected = meta_data['ap_selected']
        print(f"Processing: Loaded user_positions & ap_selected from 'user_positions.npz'.")
    except Exception as e:
        print("Error loading 'user_positions.npz':", e)
        return

    try:
        interleaved_rx = np.fromfile("rx_signal.dat", dtype=np.float32)
        rx_combined = interleaved_rx[0::2] + 1j * interleaved_rx[1::2]

        N = len(tx_baseband)
        n_users = len(user_positions)
        rx_basebands = []
        for i in range(n_users):
            start = i * N
            end = (i + 1) * N
            rx_basebands.append(rx_combined[start:end])
        print(f"Processing: Loaded RX signals for {n_users} users from 'rx_signal.dat'.")
    except Exception as e:
        print("Error loading 'rx_signal.dat':", e)
        return

    pilot_len = 64
    Fs = 1e6
    n_paths = 3
    estimations = process_channel_data(
        ap_positions,
        user_positions,
        ap_selected,
        tx_baseband,
        rx_basebands,
        pilot_len=pilot_len,
        Fs=Fs,
        n_paths=n_paths
    )

    initial_positions = []
    classic_positions = []
    Error_initial = []
    Error_classic = []

    for i, est in enumerate(estimations):
        true_pos = np.array(est['user_position'])
        ap_idx = ap_selected[i]
        ap_pos = ap_positions[ap_idx]
        DoA = est['DoA']
        DoD = est['DoD']
        DDoF = est['DDoF']
        Power = est['Power']
        weights = 10**(np.array(Power)/10)
        if np.max(weights) > 0:
            weights /= np.max(weights)
        else:
            weights = np.ones_like(weights)

        init_pos = initial_localization_with_classification(ap_pos, DoA, DDoF, weights)
        initial_positions.append(init_pos)
        Error_initial.append(np.linalg.norm(init_pos - true_pos))

        cpos = classic_localization(ap_pos, DoA, DoD, DDoF, weights)
        if cpos is not None:
            classic_positions.append(cpos)
            Error_classic.append(np.linalg.norm(cpos - true_pos))
        else:
            classic_positions.append(None)

    print(f"Median error (initial): {np.median(Error_initial)*100:.2f} cm")
    if Error_classic:
        print(f"Median error (classic): {np.median(Error_classic)*100:.2f} cm")

    virtual_anchors = find_virtual_anchors(ap_positions, estimations, initial_positions, ap_selected)
    Error_virtual = []
    virtual_positions = []
    for i, est in enumerate(estimations):
        true_pos = np.array(est['user_position'])
        ap_idx = ap_selected[i]
        if len(virtual_anchors[ap_idx]) == 0:
            virtual_positions.append(initial_positions[i])
            Error_virtual.append(Error_initial[i])
            continue
        DoA = est['DoA']
        DDoF = est['DDoF']
        Power = est['Power']
        weights = 10**(np.array(Power)/10)
        if np.max(weights) > 0:
            weights /= np.max(weights)
        else:
            weights = np.ones_like(weights)

        vpos = virtual_anchor_localization(virtual_anchors[ap_idx], DoA, DDoF, weights)
        virtual_positions.append(vpos)
        Error_virtual.append(np.linalg.norm(vpos - true_pos))

    print(f"Median error (virtual): {np.median(Error_virtual)*100:.2f} cm")

    nn_localizer = SimpleNNLocalizer()
    X, y = nn_localizer.prepare_features(estimations)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_with_anchors = []
    for i, x in enumerate(X):
        ap_idx = ap_selected[i]
        new_x = list(x)
        new_x.append(len(virtual_anchors[ap_idx]))
        if i < len(initial_positions) and initial_positions[i] is not None:
            new_x.extend(initial_positions[i])
        else:
            new_x.extend([0,0,0])
        if i < len(classic_positions) and classic_positions[i] is not None:
            new_x.extend(classic_positions[i])
        else:
            new_x.extend([0,0,0])
        if i < len(virtual_positions) and virtual_positions[i] is not None:
            new_x.extend(virtual_positions[i])
        else:
            new_x.extend([0,0,0])
        X_with_anchors.append(new_x)

    X_with_anchors = np.array(X_with_anchors)
    X_anchors_train, X_anchors_test, _, _ = train_test_split(
        X_with_anchors, y, test_size=0.2, random_state=42
    )

    # Standard NN
    print("Training standard NN model...")
    history = nn_localizer.train(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2)

    # Enhanced NN
    print("Training enhanced NN model (with anchors)...")
    nn_enhanced = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X_with_anchors.shape[1],)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(3)
    ])
    nn_enhanced.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    history_enhanced = nn_enhanced.fit(
        X_anchors_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=20, restore_best_weights=True
        )],
        verbose=1
    )

    # Evaluate
    print("Evaluating model performance...")
    metrics_standard = nn_localizer.evaluate(X_test, y_test)
    y_pred_enhanced = nn_enhanced.predict(X_anchors_test)
    position_error_enhanced = np.sqrt(np.sum((y_pred_enhanced - y_test)**2, axis=1))
    metrics_enhanced = {
        'mean_position_error': np.mean(position_error_enhanced),
        'median_position_error': np.median(position_error_enhanced),
        'position_error_90th': np.percentile(position_error_enhanced, 90),
        'position_errors': position_error_enhanced
    }

    print(f"Mean error (standard NN): {metrics_standard['mean_position_error']*100:.2f} cm")
    print(f"Median error (standard NN): {metrics_standard['median_position_error']*100:.2f} cm")
    print(f"90th percentile (standard NN): {metrics_standard['position_error_90th']*100:.2f} cm")
    print(f"Mean error (enhanced NN): {metrics_enhanced['mean_position_error']*100:.2f} cm")
    print(f"Median error (enhanced NN): {metrics_enhanced['median_position_error']*100:.2f} cm")
    print(f"90th percentile (enhanced NN): {metrics_enhanced['position_error_90th']*100:.2f} cm")

    # Plot final error CDF comparisons
    sns.set_theme(style="whitegrid", context="talk")

    plt.figure("Localization error", figsize=(10, 6))

    palette = sns.color_palette("deep")
    if Error_classic is not None:
        sorted_errors = np.sort(Error_classic)
        cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
        sns.lineplot(x=100 * sorted_errors, y=cdf, label="Weng2023 [4]", linestyle='-.', linewidth=2)
    sorted_errors = np.sort(Error_initial)
    cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    sns.lineplot(x=100 * sorted_errors, y=cdf, label="Initial", linewidth=2)

# sorted_errors = np.sort(Error_virtual)
# cdf = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
# sns.lineplot(x=100 * sorted_errors, y=cdf, label="Virtual Anchor", linestyle='--', linewidth=2)

    nn_localizer.plot_error_cdf(metrics_standard['position_errors'], label="Standard NN", linestyle='-')
    nn_localizer.plot_error_cdf(metrics_enhanced['position_errors'], label="Enhanced NN", linestyle='-.')
    plt.xlim([0, 400])
    plt.xlabel("Localization error (cm)")
    plt.ylabel("Cumulative probability (%)")
    plt.legend(loc="lower right")
    plt.title("CDF of Localization Error", fontsize=16)
    plt.tight_layout()
    plt.show()
    plt.savefig("results/error_comparison.png", dpi=300, bbox_inches='tight')

    print("Processing: Done.")

if __name__ == "__main__":
    main()
