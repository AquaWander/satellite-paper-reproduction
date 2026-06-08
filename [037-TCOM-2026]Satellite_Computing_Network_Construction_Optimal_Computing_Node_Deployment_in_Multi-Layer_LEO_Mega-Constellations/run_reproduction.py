"""
run_reproduction.py -- One-click reproduction script
Satellite Computing Network Construction: Optimal Computing Node Deployment
in Multi-Layer LEO Mega-Constellations (IEEE TCOM 2026)

Reproduces Fig.4, Fig.5, Fig.6
"""

import os
import sys
import numpy as np

# Add script directory to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

OUTPUT_DIR = os.path.join(_script_dir, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("Reproduction: Satellite Computing Network Construction")
print("IEEE Transactions on Communications, 2026")
print("=" * 60)

# ====== Step 1: Verify core formulas ======
print("\n[Step 1] Verify core formulas...")
from deployment import V_diamond, V_spot_beam

print("  Diamond coverage V_I(J):")
for J in range(1, 7):
    v = V_diamond(J)
    expected = 2 * J**2 + 2 * J + 1
    status = "OK" if v == expected else f"MISMATCH (got {v}, expected {expected})"
    print(f"    J={J}: V_I({J}) = {v}  [{status}]")

print("  Spot beam coverage V_II(J):")
for J in range(1, 7):
    v = V_spot_beam(J)
    expected_map = {1: 7, 2: 25, 3: 63, 4: 129, 5: 231, 6: 377}
    expected = expected_map[J]
    status = "OK" if v == expected else f"MISMATCH (got {v}, expected {expected})"
    print(f"    J={J}: V_II({J}) = {v}  [{status}]")

# ====== Step 2: Run simulations ======
print("\n[Step 2] Running simulations...")
from simulation import (
    simulate_nc_vs_network_size,
    simulate_delay_vs_network_size,
    simulate_delay_comparison,
    simulate_nc_comparison,
    print_simulation_summary,
)

# Fig.4 data
print("  Simulating Fig.4 data...")
fig4_nc = simulate_nc_vs_network_size(L=7)
fig4_delay = simulate_delay_vs_network_size(L=7)

# Fig.5 data
print("  Simulating Fig.5 data...")
fig5 = simulate_delay_comparison(L=7, N=50, M=50)

# Fig.6 data
print("  Simulating Fig.6 data...")
fig6 = simulate_nc_comparison(L=7)

# ====== Step 3: Print numerical summary ======
print("\n[Step 3] Numerical summary...")
print_simulation_summary(fig4_nc, fig4_delay, fig5, fig6)

# ====== Step 4: Generate figures ======
print("\n[Step 4] Generating figures...")
from plotting import plot_fig4, plot_fig5, plot_fig6

plot_fig4(fig4_nc, fig4_delay, OUTPUT_DIR)
plot_fig5(fig5, OUTPUT_DIR)
plot_fig6(fig6, OUTPUT_DIR)

# ====== Step 5: Verify results ======
print("\n[Step 5] Verifying results...")
print("\n  Expected range checks:")

# Fig.4(a): N_C should increase with N, decrease with J
for J in [1, 3, 5]:
    nc_small = fig4_nc['nc_data'][J][0]   # N=10
    nc_large = fig4_nc['nc_data'][J][-1]  # N=100
    print(f"    J={J}: N_C(N=10)={nc_small}, N_C(N=100)={nc_large}")

# Fig.4(b): delay should decrease with N (physical model)
print("\n  Fig.4(b) trend check (delay should decrease with N):")
for J in [1, 3, 5]:
    delays = fig4_delay['delay_data'][J]
    decreasing = all(delays[i] >= delays[i+1] for i in range(len(delays)-1))
    print(f"    J={J}: delay(N=10)={delays[0]:.2f} ms, delay(N=100)={delays[-1]:.2f} ms, "
          f"decreasing={decreasing}")

# Fig.5: MEO delay > LEO delay
print("\n  Fig.5 comparison:")
leo_avg = fig5['methods']['LEO Computing Node']['avg']
meo_avg = fig5['methods']['MEO Computing Node']['avg']
for i, J in enumerate(fig5['J_values']):
    print(f"    J={int(J)}: LEO_avg={leo_avg[i]:.2f} ms, MEO_avg={meo_avg[i]:.2f} ms, "
          f"MEO>LEO={meo_avg[i] > leo_avg[i]}")

# Fig.6: MEO needs fewest nodes, polygon beam needs most
print("\n  Fig.6 comparison (J=1):")
for sc_name, sc_data in fig6['scenarios'].items():
    leo_spot = sc_data['LEO Spot Beam'][0]   # J=1
    leo_poly = sc_data['LEO Polygon Beam'][0]
    meo = sc_data['MEO Computing'][0]
    print(f"    {sc_name}: LEO_spot={leo_spot}, LEO_poly={leo_poly}, MEO={meo}")

print("\n" + "=" * 60)
print(f"Reproduction complete! Figures saved to {OUTPUT_DIR}")
print("=" * 60)
