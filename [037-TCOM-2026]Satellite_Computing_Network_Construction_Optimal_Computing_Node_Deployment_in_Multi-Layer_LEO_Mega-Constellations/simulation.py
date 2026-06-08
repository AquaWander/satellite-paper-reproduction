"""
simulation.py -- Main simulation logic
Computing node count simulation, signaling distribution delay evaluation

Based on: Satellite Computing Network Construction (IEEE TCOM 2026)
Fig.4: N_C and delay vs network size and reachable hop count
Fig.5: Signaling distribution delay: LEO vs MEO computing nodes
Fig.6: Computing nodes required for different deployment schemes
"""

import numpy as np
from deployment import (
    V_spot_beam, V_diamond,
    optimal_node_count, compute_average_hops, compute_coverage_stats,
    meo_coverage_nodes, uniform_deployment_count, pso_deployment_count,
    avg_hop_diamond,
)
from config import (
    DEFAULT_L, NETWORK_SIZES, HOP_RANGE,
    PROCESSING_DELAY_MS,
    EARTH_RADIUS_KM, LIGHT_SPEED,
    DEFAULT_LEO_ALT_KM, LAYER_HEIGHT_DIFF_KM,
    MEO_ALTITUDE_KM,
)


def isl_propagation_delay_ms(N, leo_alt_km=DEFAULT_LEO_ALT_KM):
    """
    同层ISL传播延迟 (ms)
    卫星间距 = 轨道周长 / N
    延迟 = 距离 / 光速

    参数:
        N: 每轨道卫星数
        leo_alt_km: LEO轨道高度 (km)
    返回:
        单跳同层ISL传播延迟 (ms)
    """
    orbit_radius = EARTH_RADIUS_KM + leo_alt_km  # km
    orbit_circumference = 2 * np.pi * orbit_radius  # km
    isl_distance = orbit_circumference / N  # km
    delay_s = isl_distance / (LIGHT_SPEED / 1000)  # seconds (LIGHT_SPEED in m/s)
    return delay_s * 1000  # ms


def cross_layer_propagation_delay_ms(height_diff_km=LAYER_HEIGHT_DIFF_KM):
    """
    跨层ISL传播延迟 (ms)
    跨层距离 = 层间高度差
    延迟 = 距离 / 光速

    参数:
        height_diff_km: 层间高度差 (km)
    返回:
        单跳跨层ISL传播延迟 (ms)
    """
    delay_s = height_diff_km / (LIGHT_SPEED / 1000)
    return delay_s * 1000


def simulate_nc_vs_network_size(L=DEFAULT_L):
    """
    Fig.4(a) data generation
    Computing node count vs network size (N=M: 10~100) and reachable hop count J(1~6)

    Args:
        L: number of layers
    Returns:
        dict with network_sizes, J_values, nc_data, v_data
    """
    results = {
        'network_sizes': NETWORK_SIZES,
        'J_values': HOP_RANGE,
        'nc_data': {},
        'v_data': {},
    }

    for J in HOP_RANGE:
        nc_list = []
        V = V_spot_beam(J)
        results['v_data'][int(J)] = V

        for N in NETWORK_SIZES:
            M = N  # N=M
            nc = optimal_node_count(N, M, L, int(J), coverage_type='spot_beam')
            nc_list.append(nc)

        results['nc_data'][int(J)] = np.array(nc_list)

    return results


def simulate_delay_vs_network_size(L=DEFAULT_L):
    """
    Fig.4(b) data generation
    Average signaling distribution delay vs network size and reachable hop count

    Physical delay model:
    - Same-layer ISL distance = orbit_circumference / N (decreases with N)
    - Same-layer ISL propagation delay = distance / c
    - Cross-layer ISL distance = layer height difference
    - Per-hop total delay = propagation delay + processing delay
    - Total delay = avg_hops * per_hop_delay

    As N increases: satellite spacing decreases -> each hop is faster -> total delay decreases
    As J increases: avg_hops increases -> total delay increases

    Args:
        L: number of layers
    Returns:
        dict with network_sizes, J_values, delay_data
    """
    results = {
        'network_sizes': NETWORK_SIZES,
        'J_values': HOP_RANGE,
        'delay_data': {},
    }

    cross_delay = cross_layer_propagation_delay_ms()

    for J in HOP_RANGE:
        delay_list = []

        for N in NETWORK_SIZES:
            # Average hops from coverage geometry
            avg_hops = compute_average_hops(int(J))

            # Same-layer ISL propagation delay (depends on N)
            same_layer_delay = isl_propagation_delay_ms(N)

            # Weighted average: ~80% same-layer, ~20% cross-layer
            same_layer_ratio = 0.8
            cross_layer_ratio = 0.2

            per_hop_delay = (
                same_layer_ratio * same_layer_delay +
                cross_layer_ratio * cross_delay +
                PROCESSING_DELAY_MS
            )

            avg_delay = avg_hops * per_hop_delay
            delay_list.append(avg_delay)

        results['delay_data'][int(J)] = np.array(delay_list)

    return results


def simulate_delay_comparison(L=DEFAULT_L, N=50, M=50):
    """
    Fig.5 data generation
    Signaling distribution delay: LEO computing nodes vs MEO computing nodes

    Two methods:
    1. LEO computing node: lower per-hop delay (shorter ISL distance)
    2. MEO computing node: higher per-hop delay (longer LEO-MEO-LEO path)

    Each method shows max/min/avg delay across J values.

    Args:
        L: number of layers
        N, M: network dimensions
    Returns:
        dict with J_values, methods (LEO and MEO, each with max/min/avg)
    """
    methods = ['LEO Computing Node', 'MEO Computing Node']
    results = {
        'J_values': HOP_RANGE,
        'methods': {},
    }

    # LEO delay parameters
    leo_isl_delay = isl_propagation_delay_ms(N)  # same-layer ISL delay
    leo_cross_delay = cross_layer_propagation_delay_ms()  # cross-layer delay

    # MEO delay parameters
    # LEO->MEO: distance = MEO_alt - LEO_alt
    meo_distance = MEO_ALTITUDE_KM - DEFAULT_LEO_ALT_KM  # km
    meo_uplink_delay = (meo_distance / (LIGHT_SPEED / 1000)) * 1000  # ms
    # MEO processing + downlink broadcast
    meo_downlink_delay = meo_uplink_delay  # symmetric
    # MEO total per-hop: uplink + processing + downlink
    meo_per_hop_delay = meo_uplink_delay + PROCESSING_DELAY_MS + meo_downlink_delay

    for method in methods:
        max_delays = []
        min_delays = []
        avg_delays = []

        for J in HOP_RANGE:
            J = int(J)
            stats = compute_coverage_stats(J)
            avg_hops = stats['avg_hops']
            max_hops = stats['max_hops']  # = J
            # min_hops = 0: the computing node itself has 0 delay
            min_hops = 0

            if method == 'LEO Computing Node':
                # LEO: each hop is a short ISL
                per_hop_delay = leo_isl_delay + PROCESSING_DELAY_MS
                # Max delay uses the same per-hop delay (worst case: J same-layer hops)
                # This ensures max_delay >= avg_delay
                avg_delay = avg_hops * per_hop_delay
                max_delay = max_hops * per_hop_delay
                min_delay = min_hops * per_hop_delay  # = 0

            elif method == 'MEO Computing Node':
                # MEO: need to go up to MEO, process, then broadcast down
                # For a node covered by MEO computing node:
                # delay = LEO_ISL_hops * ISL_delay + uplink + MEO_processing + downlink
                # MEO covers more nodes per computing node, but each node's delay is higher
                # Average: still need some ISL hops to reach the MEO ground track
                meo_avg_hops = avg_hops * 0.5  # MEO covers wider, fewer ISL hops needed

                avg_delay = meo_avg_hops * leo_isl_delay + meo_per_hop_delay
                max_delay = max_hops * leo_isl_delay + meo_per_hop_delay
                min_delay = meo_per_hop_delay  # directly under MEO, no ISL needed

            avg_delays.append(avg_delay)
            max_delays.append(max_delay)
            min_delays.append(min_delay)

        results['methods'][method] = {
            'avg': np.array(avg_delays),
            'max': np.array(max_delays),
            'min': np.array(min_delays),
        }

    return results


def simulate_nc_comparison(L=DEFAULT_L):
    """
    Fig.6 data generation
    Computing nodes required for different deployment schemes

    Schemes:
    1. LEO spot beam (proposed): N_C = floor(N*M*L / V_II(J))
    2. LEO polygon beam: N_C = floor(N*M*L / V_I(J))
    3. MEO computing node: covers more LEO nodes per MEO satellite

    Three constellation scenarios.

    Args:
        L: default number of layers
    Returns:
        dict with J_values, scenarios
    """
    scenarios = {
        'Starlink (72x22x3)': (72, 22, 3),
        'OneWeb (18x36x3)': (18, 36, 3),
        'Large (50x50x7)': (50, 50, 7),
    }

    methods = ['LEO Spot Beam', 'LEO Polygon Beam', 'MEO Computing']

    results = {
        'J_values': HOP_RANGE,
        'scenarios': {},
    }

    for sc_name, (N, M, L_sc) in scenarios.items():
        sc_data = {}

        for method in methods:
            nc_list = []

            for J in HOP_RANGE:
                J = int(J)

                if method == 'LEO Spot Beam':
                    nc = optimal_node_count(N, M, L_sc, J, coverage_type='spot_beam')
                elif method == 'LEO Polygon Beam':
                    nc = optimal_node_count(N, M, L_sc, J, coverage_type='diamond')
                elif method == 'MEO Computing':
                    # MEO covers more LEO nodes per satellite
                    meo_cov = meo_coverage_nodes(J, N=N, M=M, L=L_sc)
                    total = N * M * L_sc
                    nc = max(1, total // meo_cov)  # at least 1 computing node

                nc_list.append(nc)

            sc_data[method] = np.array(nc_list)

        results['scenarios'][sc_name] = sc_data

    return results


def compute_energy_per_communication(distance_km, freq_ghz=30.0,
                                      G_t_db=18.0, G_r_db=18.0):
    """
    Eq.26: Free space path loss
    10*log(P_t/P_r) = 32.45 + 20*log(r*f) - G_t - G_r

    Args:
        distance_km: propagation distance (km)
        freq_ghz: frequency (GHz)
        G_t_db: transmit antenna gain (dB)
        G_r_db: receive antenna gain (dB)
    Returns:
        Path loss (dB)
    """
    if distance_km <= 0:
        return 0
    path_loss = (32.45 +
                 20 * np.log10(distance_km) +
                 20 * np.log10(freq_ghz * 1000) -  # GHz -> MHz
                 G_t_db - G_r_db)
    return path_loss


def compute_total_energy(avg_hops, hop_distance_km, cross_layer=False):
    """
    Eq.27: Total energy per communication
    E = N_retx * E_t
    Cross-layer links use 20% more energy

    Args:
        avg_hops: average hop count
        hop_distance_km: per-hop distance (km)
        cross_layer: whether cross-layer
    Returns:
        Total energy (normalized units)
    """
    from config import RETRANSMISSION_FACTOR, CROSS_LAYER_ENERGY_FACTOR

    e_single = avg_hops * RETRANSMISSION_FACTOR

    if cross_layer:
        e_single *= CROSS_LAYER_ENERGY_FACTOR

    return e_single


def print_simulation_summary(fig4_nc, fig4_delay, fig5, fig6):
    """
    Print simulation result summary for numerical verification
    """
    print("=" * 60)
    print("Simulation Results Summary")
    print("=" * 60)

    # Fig.4(a) numerical values
    print("\n--- Fig.4(a): Computing node count N_C ---")
    print(f"{'N=M':>6s}", end="")
    for J in HOP_RANGE:
        print(f"{'J='+str(int(J)):>10s}", end="")
    print()
    for i, N in enumerate(NETWORK_SIZES):
        print(f"{int(N):>6d}", end="")
        for J in HOP_RANGE:
            nc = fig4_nc['nc_data'][int(J)][i]
            print(f"{nc:>10d}", end="")
        print()

    # Fig.4(b) numerical values
    print("\n--- Fig.4(b): Average signaling distribution delay (ms) ---")
    print(f"{'N=M':>6s}", end="")
    for J in HOP_RANGE:
        print(f"{'J='+str(int(J)):>10s}", end="")
    print()
    for i, N in enumerate(NETWORK_SIZES):
        print(f"{int(N):>6d}", end="")
        for J in HOP_RANGE:
            d = fig4_delay['delay_data'][int(J)][i]
            print(f"{d:>10.2f}", end="")
        print()

    # Verify Fig.4(b) trend: delay should decrease with N
    print("\n  [Trend check] Delay decreasing with N?")
    for J in HOP_RANGE:
        delays = fig4_delay['delay_data'][int(J)]
        is_decreasing = all(delays[i] >= delays[i+1] for i in range(len(delays)-1))
        print(f"    J={int(J)}: delay(N=10)={delays[0]:.2f}, delay(N=100)={delays[-1]:.2f}, "
              f"decreasing={is_decreasing}")

    # Fig.5 numerical values
    print("\n--- Fig.5: Signaling distribution delay comparison (N=M=50, L=7) ---")
    for method, data in fig5['methods'].items():
        print(f"  {method}:")
        for i, J in enumerate(HOP_RANGE):
            print(f"    J={int(J)}: avg={data['avg'][i]:.2f}, "
                  f"max={data['max'][i]:.2f}, min={data['min'][i]:.2f} ms")
            # Verify min <= avg <= max
            if not (data['min'][i] <= data['avg'][i] <= data['max'][i]):
                print(f"    WARNING: min<=avg<=max violated!")

    # Fig.6 numerical values
    print("\n--- Fig.6: Computing nodes for different schemes ---")
    for sc_name, sc_data in fig6['scenarios'].items():
        print(f"  {sc_name}:")
        for method, nc_arr in sc_data.items():
            vals = ", ".join([f"J={int(J)}:{nc_arr[i]}" for i, J in enumerate(HOP_RANGE)])
            print(f"    {method}: {vals}")

    # V(J) reference values
    print("\n--- Coverage node count V(J) reference ---")
    for J in HOP_RANGE:
        J = int(J)
        vi = V_diamond(J)
        vii = V_spot_beam(J)
        print(f"  J={J}: V_diamond(J)={vi}, V_spot_beam(J)={vii}")

    # Table III verification
    # Note: Table III uses V_diamond (V_I) for N_C calculation, not V_spot_beam (V_II)
    # This is confirmed by matching all expected values
    print("\n--- Table III verification (Starlink 3-layer) ---")
    print("  (Uses V_diamond / V_I for coverage)")
    starlink_layers = [(9, 277), (42, 59), (9, 283)]
    total_starlink = sum(N_l * M_l for N_l, M_l in starlink_layers)
    print(f"  Starlink total: {total_starlink} (expected: 7518)")
    for J in range(1, 8):
        V = V_diamond(J)
        nc = total_starlink // V
        expected = {1: 1503, 2: 578, 3: 300, 4: 183, 5: 123, 6: 88, 7: 66}
        match = "OK" if nc == expected.get(J, nc) else f"MISMATCH (expected {expected.get(J, '?')})"
        print(f"  J={J}: N_C = {total_starlink}/{V} = {nc}  [{match}]")

    print("\n--- Table III verification (GW-A59) ---")
    print("  (Uses V_diamond / V_I for coverage)")
    gw_layers = [(16, 30), (40, 50), (60, 60)]
    total_gw = sum(N_l * M_l for N_l, M_l in gw_layers)
    print(f"  GW-A59 total: {total_gw} (expected: 6080)")
    for J in range(1, 8):
        V = V_diamond(J)
        nc = total_gw // V
        expected = {1: 1216, 2: 467, 3: 243, 4: 148, 5: 99, 6: 71, 7: 53}
        match = "OK" if nc == expected.get(J, nc) else f"MISMATCH (expected {expected.get(J, '?')})"
        print(f"  J={J}: N_C = {total_gw}/{V} = {nc}  [{match}]")

    print("\n--- Table III verification (OneWeb) ---")
    print("  (Uses V_diamond / V_I for coverage)")
    oneweb_total = 18 * 40
    print(f"  OneWeb total: {oneweb_total} (expected: 720)")
    for J in range(1, 8):
        V = V_diamond(J)
        nc = oneweb_total // V
        expected = {1: 144, 2: 55, 3: 28, 4: 17, 5: 11, 6: 8, 7: 6}
        match = "OK" if nc == expected.get(J, nc) else f"MISMATCH (expected {expected.get(J, '?')})"
        print(f"  J={J}: N_C = {oneweb_total}/{V} = {nc}  [{match}]")

    print("\n" + "=" * 60)
