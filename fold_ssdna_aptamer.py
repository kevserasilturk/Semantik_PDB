#!/usr/bin/env python3
"""
================================================================================
 GPU-Accelerated ssDNA Aptamer Folding Pipeline
================================================================================
 Folds a 75-nucleotide single-stranded DNA (ssDNA) aptamer into a stable 3D
 tertiary structure (.pdb) using GPU-accelerated molecular dynamics via OpenMM.

 Pipeline:
   Step 1 - Parse hardcoded secondary structure (dot-bracket)
   Step 2 - Build initial linear ssDNA chain (NeRF + idealized geometry)
   Step 3 - AMBER14 force field + OBC2 implicit solvent + base-pair restraints
   Step 4 - GPU simulated annealing  (heat -> cool -> equilibrate)
   Step 5 - Restraint removal, final minimization, PDB output

 Dependencies:  pip install openmm biopython numpy
 GPU:           NVIDIA GPU (tries CUDA, then OpenCL, then CPU)
================================================================================
"""

import os, sys, gc, time, math, warnings, traceback, logging
from datetime import datetime
import numpy as np

warnings.filterwarnings("ignore", category=DeprecationWarning)

# ===========================================================================
#  LOGGING
# ===========================================================================
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("folding_pipeline.log", mode="w"),
    ],
)
log = logging.getLogger("AptamerFold")

# ===========================================================================
#  CONFIGURATION - DNA SEQUENCE & SECONDARY STRUCTURE
# ===========================================================================
# You can easily add and swap new DNA sequences and folds here.
# Just change the ACTIVE TARGET variables below.

# --- ACTIVE TARGET ---
SEQUENCE = "UAGCUUAUCAGACUGAUGUUGA"

# +----------------------------------------------------------------------+
# |  PASTE your RNAfold/MFE dot-bracket on the next line.                |
# |  Length MUST match the SEQUENCE length.                              |
# +----------------------------------------------------------------------+
DOT_BRACKET = "......((((...))))....."

# --- PAST / ALTERNATIVE TARGETS (For easy swapping) ---
# SEQUENCE_1 = "GCAATGGTACGGTACTTCCTGAATGTTGTTTTTTCTCTTTTCTCTATAGTACAAAAGTGCACGCTACTTTGCTAA"
# DOT_BRACKET_1 = "((((..(((((((((((........((((.....................)))))))))).)).))).))))..."

# Files
OUTPUT_PDB      = "Mi-Rna.pdb"
INITIAL_PDB     = "initial_linear_ssdna.pdb"
FIXED_PDB       = "fixed_ssdna.pdb"
CHECKPOINT_DIR  = "checkpoints"

# Simulation knobs
HEATING_STEPS             = 500_000
COOLING_STEPS             = 5_000_000
EQUILIBRATION_STEPS       = 1_000_000
RESTRAINT_REMOVAL_STEPS   = 500_000
REPORT_INTERVAL           = 50_000
TIMESTEP_FS               = 1.0
RESTRAINT_K               = 100.0        # kJ/mol/nm^2

T_START = 300.0
T_HIGH  = 370.0
T_FINAL = 300.0

# ===========================================================================
#  STEP 1 - PARSE DOT-BRACKET
# ===========================================================================
def parse_dot_bracket(sequence: str, dot_bracket: str) -> list:
    """Return sorted list of (i, j) base-pair tuples from dot-bracket."""
    log.info("=" * 70)
    log.info("STEP 1: Parsing secondary structure")
    log.info("=" * 70)
    assert len(sequence) == len(dot_bracket), (
        f"Length mismatch: seq={len(sequence)} db={len(dot_bracket)}"
    )
    pairs, stack = [], []
    for idx, ch in enumerate(dot_bracket):
        if ch == "(":
            stack.append(idx)
        elif ch == ")":
            if not stack:
                raise ValueError(f"Unmatched ')' at {idx}")
            pairs.append((stack.pop(), idx))
        elif ch != ".":
            raise ValueError(f"Bad char '{ch}' at {idx}")
    if stack:
        raise ValueError(f"Unmatched '(' at {stack}")
    pairs.sort()
    log.info(f"  Sequence length : {len(sequence)}")
    log.info(f"  Base pairs found: {len(pairs)}")
    for a, b in pairs:
        log.info(f"    {sequence[a]}{a+1} -- {sequence[b]}{b+1}")
    if not pairs:
        log.warning("  *** NO BASE PAIRS - structure will stay linear! ***")
        log.warning("  *** Paste your MFE dot-bracket into DOT_BRACKET ***")
    return pairs

# ===========================================================================
#  STEP 2 - BUILD INITIAL LINEAR ssDNA
# ===========================================================================

# ---- NeRF (Natural Extension Reference Frame) ----
def nerf(a, b, c, bond_len, angle_rad, dihed_rad):
    """Place atom D from three ancestors + internal coords (radians)."""
    bc = c - b
    n_bc = np.linalg.norm(bc)
    bc = bc / n_bc if n_bc > 1e-12 else np.array([1., 0., 0.])

    ab = b - a
    n_ab = np.linalg.norm(ab)
    ab = ab / n_ab if n_ab > 1e-12 else np.array([0., 1., 0.])

    n = np.cross(ab, bc)
    nn = np.linalg.norm(n)
    if nn < 1e-12:
        perp = np.array([1, 0, 0]) if abs(bc[0]) < 0.9 else np.array([0, 1, 0])
        n = np.cross(bc, perp)
    n = n / np.linalg.norm(n)
    nbc = np.cross(n, bc)

    sa = math.sin(math.pi - angle_rad)
    ca = math.cos(math.pi - angle_rad)
    return c + bond_len * (bc * ca + nbc * sa * math.cos(dihed_rad) + n * sa * math.sin(dihed_rad))

def _r(deg):
    """Degrees to radians."""
    return math.radians(deg)

# Bond lengths (Angstroms)
BL = dict(
    P_O5=1.593, O5_C5=1.440, C5_C4=1.510, C4_O4=1.451,
    C4_C3=1.526, C3_O3=1.423, O3_P=1.607, C3_C2=1.523,
    C2_C1=1.529, C1_O4=1.412, C1_N=1.475, P_OP=1.485,
)
# Bond angles (degrees)
BA = dict(
    O3_P_O5=104.0, P_O5_C5=120.9, O5_C5_C4=111.0, C5_C4_C3=116.4,
    C5_C4_O4=109.5, C4_C3_O3=119.7, C4_C3_C2=102.4, C3_C2_C1=102.0,
    C2_C1_O4=106.0, C4_O4_C1=109.5, C3_O3_P=119.7, O4_C1_N=108.1,
    O5_P_OP=109.5,
)

# Base heavy-atom templates (planar, origin at N9/N1)
BASE_TEMPLATES = {
    "A": {
        "N9": np.array([ 0.00, 0.00, 0.0]),
        "C8": np.array([ 1.12, 0.79, 0.0]),
        "N7": np.array([ 0.82, 2.06, 0.0]),
        "C5": np.array([-0.55, 2.06, 0.0]),
        "C6": np.array([-1.47, 3.10, 0.0]),
        "N6": np.array([-1.04, 4.38, 0.0]),
        "N1": np.array([-2.80, 2.82, 0.0]),
        "C2": np.array([-3.15, 1.54, 0.0]),
        "N3": np.array([-2.33, 0.50, 0.0]),
        "C4": np.array([-1.02, 0.80, 0.0]),
    },
    "G": {
        "N9": np.array([ 0.00, 0.00, 0.0]),
        "C8": np.array([ 1.12, 0.79, 0.0]),
        "N7": np.array([ 0.82, 2.06, 0.0]),
        "C5": np.array([-0.55, 2.06, 0.0]),
        "C6": np.array([-1.47, 3.10, 0.0]),
        "O6": np.array([-1.22, 4.31, 0.0]),
        "N1": np.array([-2.80, 2.66, 0.0]),
        "C2": np.array([-3.17, 1.34, 0.0]),
        "N2": np.array([-4.47, 1.12, 0.0]),
        "N3": np.array([-2.33, 0.31, 0.0]),
        "C4": np.array([-1.02, 0.75, 0.0]),
    },
    "T": {
        "N1": np.array([ 0.00, 0.00, 0.0]),
        "C6": np.array([ 1.20,-0.67, 0.0]),
        "C5": np.array([ 1.23,-2.04, 0.0]),
        "C7": np.array([ 2.49,-2.82, 0.0]),
        "C4": np.array([ 0.01,-2.75, 0.0]),
        "O4": np.array([ 0.01,-3.98, 0.0]),
        "N3": np.array([-1.16,-2.01, 0.0]),
        "C2": np.array([-1.18,-0.67, 0.0]),
        "O2": np.array([-2.23,-0.02, 0.0]),
    },
    "C": {
        "N1": np.array([ 0.00, 0.00, 0.0]),
        "C6": np.array([ 1.20,-0.67, 0.0]),
        "C5": np.array([ 1.23,-2.04, 0.0]),
        "C4": np.array([ 0.01,-2.75, 0.0]),
        "N4": np.array([ 0.01,-4.09, 0.0]),
        "N3": np.array([-1.16,-2.01, 0.0]),
        "C2": np.array([-1.18,-0.67, 0.0]),
        "O2": np.array([-2.27,-0.02, 0.0]),
    },
    "U": {
        "N1": np.array([ 0.00, 0.00, 0.0]),
        "C6": np.array([ 1.20,-0.67, 0.0]),
        "C5": np.array([ 1.23,-2.04, 0.0]),
        "C4": np.array([ 0.01,-2.75, 0.0]),
        "O4": np.array([ 0.01,-3.98, 0.0]),
        "N3": np.array([-1.16,-2.01, 0.0]),
        "C2": np.array([-1.18,-0.67, 0.0]),
        "O2": np.array([-2.23,-0.02, 0.0]),
    },
}
PURINES = {"A", "G"}

def _amber_resname(base, idx, total, is_rna=False):
    if is_rna:
        m = {"G": "G", "C": "C", "A": "A", "U": "U"}
    else:
        m = {"G": "DG", "C": "DC", "A": "DA", "T": "DT"}
    n = m.get(base, base)
    if idx == 0:        n += "5"
    elif idx == total-1: n += "3"
    return n


def build_linear_ssdna(sequence, output_pdb):
    """Build extended single-strand DNA/RNA chain via NeRF, write PDB."""
    log.info("=" * 70)
    is_rna = "U" in sequence
    mol_type = "RNA" if is_rna else "ssDNA"
    log.info(f"STEP 2: Building initial linear {mol_type} chain")
    log.info("=" * 70)

    n = len(sequence)
    # Extended-backbone torsions (°)
    ALPHA, BETA, GAMMA  = -60.0, 180.0, 60.0
    DELTA, EPSLN, ZETA  = 143.0, 180.0, -85.0
    CHI                  = -120.0          # anti

    all_atoms = []
    ser: int = 1

    # seed predecessors
    pp = np.array([0., 0., -3.0])
    pv = np.array([0., 0., -1.5])
    o3_prev = np.array([0., 0., 0.])

    for i, base in enumerate(sequence):
        rn   = _amber_resname(base, i, n, is_rna)
        rid  = i + 1
        is5  = (i == 0)
        nuc  = {}                       # atom_name -> xyz

        # -- backbone ------------------------------------------
        if not is5:
            p = nerf(pp, pv, o3_prev, BL["O3_P"],  _r(BA["C3_O3_P"]), _r(ZETA))
            nuc["P"]   = p
            nuc["OP1"] = nerf(pv, o3_prev, p, BL["P_OP"], _r(BA["O5_P_OP"]), _r(120))
            nuc["OP2"] = nerf(pv, o3_prev, p, BL["P_OP"], _r(BA["O5_P_OP"]), _r(-120))
            o5 = nerf(pv, o3_prev, p, BL["P_O5"], _r(BA["O3_P_O5"]), _r(ALPHA))
            nuc["O5'"] = o5
            anc_a, anc_b = o3_prev, p
        else:
            o5 = np.array([0., 1.0, i * 6.5])
            nuc["O5'"] = o5
            anc_a, anc_b = pp, pv

        c5 = nerf(anc_a, anc_b, o5, BL["O5_C5"], _r(BA["P_O5_C5"]),  _r(BETA))
        nuc["C5'"] = c5
        c4 = nerf(anc_b, o5, c5,   BL["C5_C4"],  _r(BA["O5_C5_C4"]), _r(GAMMA))
        nuc["C4'"] = c4
        o4 = nerf(o5, c5, c4,      BL["C4_O4"],  _r(BA["C5_C4_O4"]), _r(120))
        nuc["O4'"] = o4
        c3 = nerf(o5, c5, c4,      BL["C4_C3"],  _r(BA["C5_C4_C3"]), _r(DELTA))
        nuc["C3'"] = c3
        o3 = nerf(c5, c4, c3,      BL["C3_O3"],  _r(BA["C4_C3_O3"]), _r(EPSLN))
        nuc["O3'"] = o3
        c2 = nerf(c5, c4, c3,      BL["C3_C2"],  _r(BA["C4_C3_C2"]), _r(-120))
        nuc["C2'"] = c2
        c1 = nerf(c4, c3, c2,      BL["C2_C1"],  _r(BA["C3_C2_C1"]), _r(30))
        nuc["C1'"] = c1

        if is_rna:
            o2 = nerf(c1, c3, c2, 1.41, _r(110.0), _r(120.0))
            nuc["O2'"] = o2

        # -- base ----------------------------------------------
        gly = "N9" if base in PURINES else "N1"
        n_base = nerf(c3, c2, c1, BL["C1_N"], _r(BA["O4_C1_N"]), _r(CHI))
        nuc[gly] = n_base

        tmpl  = BASE_TEMPLATES[base]
        g_vec = n_base - c1
        g_vec = g_vec / np.linalg.norm(g_vec)
        v1    = c4 - c1
        bn    = np.cross(g_vec, v1)
        bnn   = np.linalg.norm(bn)
        bn    = bn / bnn if bnn > 1e-12 else np.array([0, 0, 1.])
        v3    = np.cross(bn, g_vec); v3 = v3 / np.linalg.norm(v3)
        rot   = np.column_stack([g_vec, v3, bn])
        orig  = tmpl[gly]
        for aname, tc in tmpl.items():
            if aname == gly:
                continue
            nuc[aname] = n_base + rot @ (tc - orig)

        # collect
        for aname, xyz in nuc.items():
            elem = aname[0] if aname[0] in "CNOP" else aname[0]
            all_atoms.append((ser, aname, rn, "A", rid,
                              xyz[0], xyz[1], xyz[2], elem))
            ser += 1  # type: ignore

        pp, pv, o3_prev = c4, c3, o3

    # -- write PDB ---------------------------------------------
    log.info(f"  Writing {len(all_atoms)} heavy atoms -> {output_pdb}")
    with open(output_pdb, "w") as f:
        f.write(f"REMARK   Nucleic Acid Pipeline - initial extended chain\n")
        f.write(f"REMARK   {datetime.now().isoformat()}\n")
        for (s, an, rn, ch, ri, x, y, z, el) in all_atoms:
            an4 = f" {an:<3s}" if len(an) < 4 else an
            f.write(f"ATOM  {s:5d} {an4:4s} {rn:<4s}{ch:1s}{ri:4d}    "
                    f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n")
        f.write("TER\nEND\n")
    log.info("  [OK] Initial PDB written")
    return output_pdb


# ===========================================================================
#  STEP 3 - FORCE FIELD  +  RESTRAINTS  +  SIMULATION OBJECT
# ===========================================================================
def setup_simulation(initial_pdb, base_pairs, sequence):
    """
    Load PDB -> Modeller.addHydrogens -> AMBER14/OBC2 system -> restraints ->
    GPU simulation.
    Returns (simulation, system, bp_force, posres_force, posres_force_index).
    """
    log.info("=" * 70)
    log.info("STEP 3: Force field, restraints, GPU simulation")
    log.info("=" * 70)

    from openmm.app import (PDBFile, ForceField, Modeller, Simulation,
                             HBonds, NoCutoff, PDBReporter)
    from openmm import (Platform, LangevinMiddleIntegrator,
                        CustomBondForce, CustomExternalForce,
                        LocalEnergyMinimizer)
    from openmm.unit import (nanometer, nanometers, kelvin,
                             picosecond, picoseconds, femtoseconds,
                             kilojoules_per_mole, angstroms)

    # -- 3a  load PDB -----------------------------------------
    log.info("  Loading initial PDB...")
    pdb = PDBFile(initial_pdb)

    # -- 3b  add hydrogens via Modeller ------------------------
    log.info("  Loading AMBER14 + OBC2 force field...")
    ff  = ForceField("amber14-all.xml", "implicit/obc2.xml")

    log.info("  Adding hydrogens with Modeller (pH 7.0)...")
    modeller = Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens(ff, pH=7.0)

    topology  = modeller.topology
    positions = modeller.positions

    # save fixed PDB
    with open(FIXED_PDB, "w") as fh:
        PDBFile.writeFile(topology, positions, fh)
    log.info(f"  Fixed PDB saved -> {FIXED_PDB}")

    n_atoms = topology.getNumAtoms()
    log.info(f"  Total atoms (with H): {n_atoms}")

    # -- 3c  create system -------------------------------------
    log.info("  Creating OpenMM System...")
    system = ff.createSystem(topology,
                             nonbondedMethod=NoCutoff,
                             constraints=HBonds)
    log.info(f"  System particles: {system.getNumParticles()}")

    # -- 3d  base-pair harmonic restraints ---------------------
    bp_force = None
    if base_pairs:
        log.info(f"  Adding {len(base_pairs)} base-pair restraints...")
        bp_force = CustomBondForce("0.5*k*(r-r0)^2")
        bp_force.addPerBondParameter("k")
        bp_force.addPerBondParameter("r0")

        # atom lookup  (residue_index, atom_name) -> particle index
        lut = {}
        for atom in topology.atoms():
            lut[(atom.residue.index, atom.name)] = atom.index

        pair_map = {
            ("A","T"): ("N1","N3", 0.282),
            ("T","A"): ("N3","N1", 0.282),
            ("G","C"): ("N1","N3", 0.291),
            ("C","G"): ("N3","N1", 0.291),
            ("G","T"): ("O6","O4", 0.280),
            ("T","G"): ("O4","O6", 0.280),
            ("A","U"): ("N1","N3", 0.282),
            ("U","A"): ("N3","N1", 0.282),
            ("G","U"): ("O6","O4", 0.280),
            ("U","G"): ("O4","O6", 0.280),
        }
        added: int = 0
        for (i, j) in base_pairs:
            key = (sequence[i], sequence[j])
            if key not in pair_map:
                log.warning(f"    Skip unusual pair {sequence[i]}{i+1}-{sequence[j]}{j+1}")
                continue
            a_i, a_j, r0 = pair_map[key]
            idx_i = lut.get((i, a_i))
            idx_j = lut.get((j, a_j))
            if idx_i is None or idx_j is None:
                log.warning(f"    Atoms not found for pair {i+1}-{j+1}")
                continue
            bp_force.addBond(idx_i, idx_j, [RESTRAINT_K, r0])
            added += 1  # type: ignore
            log.info(f"    {sequence[i]}{i+1}:{a_i} <-> {sequence[j]}{j+1}:{a_j}  "
                     f"r0={r0*10:.1f}A  k={RESTRAINT_K}")
        bp_idx = system.addForce(bp_force)
        log.info(f"  {added} restraints added (force #{bp_idx})")
    else:
        log.warning("  No base pairs -> no restraints")

    # -- 3e  position restraints (for staged minimization) -----
    log.info("  Adding position restraints (staged-min safety net)...")
    posres = CustomExternalForce("k_pos*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    posres.addGlobalParameter("k_pos", 1000.0)
    posres.addPerParticleParameter("x0")
    posres.addPerParticleParameter("y0")
    posres.addPerParticleParameter("z0")
    for idx in range(system.getNumParticles()):
        p = positions[idx]
        posres.addParticle(idx, [p.x, p.y, p.z])
    posres_idx = system.addForce(posres)

    # -- 3f  select GPU platform -------------------------------
    log.info("  Selecting compute platform...")
    platform, props = None, {}
    for pname in ("CUDA", "OpenCL", "CPU"):
        try:
            platform = Platform.getPlatformByName(pname)
            if pname == "CUDA":
                props = {"DeviceIndex": "0", "Precision": "mixed"}
            elif pname == "OpenCL":
                props = {"OpenCLPlatformIndex": "0",
                         "DeviceIndex": "0",
                         "Precision": "mixed"}
            log.info(f"  [OK] Using {pname}  (speed {platform.getSpeed():.0f})")
            break
        except Exception:
            log.info(f"  [--] {pname} unavailable")
    if platform is None:
        raise RuntimeError("No compute platform found")

    # -- 3g  integrator & simulation ---------------------------
    integrator = LangevinMiddleIntegrator(
        T_START * kelvin, 10.0 / picosecond, TIMESTEP_FS * femtoseconds)
    sim = Simulation(topology, system, integrator, platform, props)
    sim.context.setPositions(positions)
    log.info(f"  Simulation ready -- {n_atoms} atoms on {platform.getName()}")
    gc.collect()
    return sim, system, bp_force, posres, posres_idx


# ===========================================================================
#  STEP 3.5 - STAGED MINIMIZATION  (NaN prevention)
# ===========================================================================
def staged_minimization(sim, system, posres, posres_idx):
    """
    Multi-stage minimization with decreasing position-restraint strength
    and decreasing tolerance to tame initial steric clashes.
    """
    from openmm import LocalEnergyMinimizer
    from openmm.unit import kilojoules_per_mole, nanometer

    log.info("=" * 70)
    log.info("STEP 3.5: Staged energy minimization (NaN prevention)")
    log.info("=" * 70)

    stages = [
        # (k_pos, tolerance, max_iter, label)
        (1000.0, 50000.0,  500,   "Strong restraints, coarse tol"),
        (500.0,  10000.0,  1000,  "Moderate-strong restraints"),
        (100.0,  5000.0,   2000,  "Medium restraints"),
        (10.0,   500.0,    5000,  "Weak restraints"),
        (1.0,    50.0,     10000, "Very weak restraints"),
        (0.0,    10.0,     20000, "Free pass 1"),
        (0.0,    1.0,      50000, "Free pass 2 (deep)"),
    ]

    for k, tol, mi, label in stages:
        log.info(f"  [{label}]  k_pos={k:.0f}  tol={tol:.0f}  max_iter={mi}")
        sim.context.setParameter("k_pos", k)

        # pre-check energy
        st = sim.context.getState(getEnergy=True, getPositions=True)
        pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        if math.isnan(pe) or math.isinf(pe):
            log.warning(f"    Energy={pe} -> perturbing coordinates...")
            pos = st.getPositions(asNumpy=True)
            noise = np.random.normal(0, 0.005, pos.shape) * nanometer
            sim.context.setPositions(pos + noise)

        try:
            LocalEnergyMinimizer.minimize(sim.context, tol, mi)
        except Exception as e:
            log.error(f"    minimize failed: {e} -- trying 10x tol")
            try:
                LocalEnergyMinimizer.minimize(sim.context, tol * 10, mi // 2)
            except Exception as e2:
                log.error(f"    second attempt failed: {e2}")
                continue

        st = sim.context.getState(getEnergy=True)
        pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        tag = "[OK]" if not (math.isnan(pe) or math.isinf(pe)) else "[FAIL NaN/Inf!]"
        log.info(f"    {tag}  PE = {pe:.1f} kJ/mol")
        gc.collect()

    # -- convergence loop: keep minimizing until energy stabilizes --
    log.info("  Convergence loop (until energy change < 1%)...")
    prev_pe: float = float('inf')
    for rnd in range(20):  # max 20 rounds
        try:
            LocalEnergyMinimizer.minimize(sim.context, 1.0, 100000)
        except Exception:
            break
        st = sim.context.getState(getEnergy=True)
        pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        if math.isnan(pe) or math.isinf(pe):
            break
        change = abs(prev_pe - pe) / (abs(pe) + 1e-10)  # type: ignore
        log.info(f"    Round {rnd+1}: PE = {pe:.1f} kJ/mol  (change={change*100:.2f}%)")
        if change < 0.01:  # less than 1% change
            log.info("    Converged!")
            break
        prev_pe = pe
        gc.collect()

    st = sim.context.getState(getEnergy=True)
    pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    log.info(f"  Final post-minimization energy: {pe:.1f} kJ/mol")
    return pe


# ===========================================================================
#  helpers
# ===========================================================================
_last_good_positions = [None]  # mutable container for last non-NaN positions

def _save_checkpoint(sim, path):
    """Save checkpoint. Uses last good positions if current ones are NaN."""
    from openmm.app import PDBFile
    from openmm.unit import kilojoules_per_mole
    try:
        st = sim.context.getState(getPositions=True, getEnergy=True)
        pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
        pos = st.getPositions()
        if not math.isnan(pe):
            _last_good_positions[0] = pos
            with open(path, "w") as f:
                PDBFile.writeFile(sim.topology, pos, f)
            return True
    except Exception:
        pass
    # fallback: use last good positions
    if _last_good_positions[0] is not None:
        log.warning(f"  Using last good positions for checkpoint {path}")
        with open(path, "w") as f:
            PDBFile.writeFile(sim.topology, _last_good_positions[0], f)
        return True
    log.error(f"  Cannot save checkpoint {path} -- no good positions available")
    return False


# ===========================================================================
#  STEP 4 - SIMULATED ANNEALING
# ===========================================================================
def simulated_annealing(sim, system, base_pairs):
    from openmm.unit import kelvin, kilojoules_per_mole

    log.info("=" * 70)
    log.info("STEP 4: GPU simulated annealing")
    log.info("=" * 70)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    def _run_phase(label, n_steps, t_start, t_end):
        """Run a temperature-ramp phase. Returns False on NaN."""
        log.info(f"\n  -- {label}: {t_start:.0f}K -> {t_end:.0f}K  "
                 f"({n_steps:,} steps) --")
        t0 = time.time()
        intervals = n_steps // REPORT_INTERVAL
        for iv in range(intervals):
            frac = (iv + 1) / intervals
            temp = t_start + frac * (t_end - t_start)
            sim.integrator.setTemperature(temp * kelvin)

            # Save last good state BEFORE stepping
            try:
                st_pre = sim.context.getState(getPositions=True, getEnergy=True)
                pe_pre = st_pre.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
                if not math.isnan(pe_pre):
                    _last_good_positions[0] = st_pre.getPositions()
            except Exception:
                pass

            try:
                sim.step(REPORT_INTERVAL)
            except Exception as e:
                log.error(f"  Crash at step {(iv+1)*REPORT_INTERVAL}: {e}")
                return False
            st = sim.context.getState(getEnergy=True)
            pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            step = (iv + 1) * REPORT_INTERVAL
            if math.isnan(pe):
                log.error(f"  NaN at step {step}!")
                return False
            log.info(f"  {step:>10,}/{n_steps:,} | T={temp:6.1f}K | "
                     f"PE={pe:>14.1f} kJ/mol")
            # checkpoint every 10 reports
            if step % (REPORT_INTERVAL * 10) == 0:
                _save_checkpoint(sim, os.path.join(
                    CHECKPOINT_DIR, f"{label.lower().replace(' ','_')}_{step}.pdb"))
            if step % 500_000 == 0:
                gc.collect()
        log.info(f"  {label} done in {time.time()-t0:.0f}s")
        return True

    ok = _run_phase("Heating",       HEATING_STEPS,       T_START, T_HIGH)
    if not ok:
        return False
    ok = _run_phase("Cooling",       COOLING_STEPS,       T_HIGH, T_FINAL)
    if not ok:
        return False
    ok = _run_phase("Equilibration", EQUILIBRATION_STEPS, T_FINAL, T_FINAL)
    return ok


# ===========================================================================
#  STEP 5 - FINAL RELAXATION  +  OUTPUT
# ===========================================================================
def final_relaxation(sim, system, bp_force, posres, posres_idx, base_pairs):
    from openmm import LocalEnergyMinimizer
    from openmm.unit import kilojoules_per_mole

    log.info("=" * 70)
    log.info("STEP 5: Final relaxation & output")
    log.info("=" * 70)

    # -- 5a  ramp base-pair restraints to zero -----------------
    if bp_force is not None and base_pairs:
        log.info(f"  Removing restraints over {RESTRAINT_REMOVAL_STEPS:,} steps...")
        n_iv = 10
        steps_per = RESTRAINT_REMOVAL_STEPS // n_iv
        for iv in range(n_iv):
            frac = 1.0 - (iv + 1) / n_iv
            new_k = RESTRAINT_K * frac
            for bi in range(bp_force.getNumBonds()):
                p1, p2, params = bp_force.getBondParameters(bi)
                bp_force.setBondParameters(bi, p1, p2, [new_k, params[1]])
            bp_force.updateParametersInContext(sim.context)
            sim.step(steps_per)
            st = sim.context.getState(getEnergy=True)
            pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
            log.info(f"    k={new_k:6.1f} | PE={pe:.1f} kJ/mol")
            gc.collect()

    sim.context.setParameter("k_pos", 0.0)

    # -- 5b  final minimization --------------------------------
    log.info("  Final energy minimization (restraint-free)...")
    try:
        LocalEnergyMinimizer.minimize(sim.context, 10.0, 10000)
    except Exception:
        log.warning("  Fine min failed -- trying coarse...")
        try:
            LocalEnergyMinimizer.minimize(sim.context, 100.0, 5000)
        except Exception:
            log.error("  Could not minimize — saving current state")

    st = sim.context.getState(getEnergy=True, getPositions=True)
    pe = st.getPotentialEnergy().value_in_unit(kilojoules_per_mole)
    log.info(f"  Final energy: {pe:.1f} kJ/mol")

    # -- 5c  write output PDB ---------------------------------
    from openmm.app import PDBFile
    with open(OUTPUT_PDB, "w") as fh:
        PDBFile.writeFile(sim.topology, st.getPositions(), fh)
    log.info(f"  [OK] Output saved -> {OUTPUT_PDB}")

    _save_checkpoint(sim, os.path.join(CHECKPOINT_DIR, "final.pdb"))
    return pe


# ===========================================================================
#  MAIN
# ===========================================================================
def main():
    t0 = time.time()
    log.info("+" + "="*68 + "+")
    log.info("|  ssDNA / RNA Aptamer Folding Pipeline -- GPU Accelerated           |")
    log.info("+" + "="*68 + "+")
    log.info(f"  Time     : {datetime.now().isoformat()}")
    log.info(f"  Sequence : {SEQUENCE}")
    log.info(f"  Length   : {len(SEQUENCE)} nt")
    log.info(f"  Output   : {OUTPUT_PDB}")

    if DOT_BRACKET == "." * len(SEQUENCE):
        log.warning("+" + "="*68 + "+")
        log.warning("|  DOT_BRACKET is all dots -> NO folding will occur!               |")
        log.warning("|  Paste your RNAfold MFE result into the DOT_BRACKET variable.   |")
        log.warning("+" + "="*68 + "+")

    try:
        # 1) parse structure
        bp = parse_dot_bracket(SEQUENCE, DOT_BRACKET)
        gc.collect()

        # 2) build chain
        build_linear_ssdna(SEQUENCE, INITIAL_PDB)
        gc.collect()

        # 3) setup simulation
        sim, system, bp_force, posres, posres_idx = \
            setup_simulation(INITIAL_PDB, bp, SEQUENCE)
        gc.collect()

        # 3.5) staged minimization
        staged_minimization(sim, system, posres, posres_idx)
        _save_checkpoint(sim, os.path.join(CHECKPOINT_DIR, "post_min.pdb"))
        gc.collect()

        # 4) simulated annealing
        sim.context.setParameter("k_pos", 0.0)   # release pos restraints
        ok = simulated_annealing(sim, system, bp)
        if not ok:
            log.error("Annealing failed -- saving last good state")
            _save_checkpoint(sim, OUTPUT_PDB)
        else:
            # 5) final relaxation
            final_relaxation(sim, system, bp_force, posres, posres_idx, bp)

    except Exception:
        log.critical(f"Pipeline error:\n{traceback.format_exc()}")
        sys.exit(1)
    finally:
        gc.collect()

    elapsed = time.time() - t0
    h, rem = divmod(int(elapsed), 3600)
    m, s   = divmod(rem, 60)
    log.info("")
    log.info("+" + "="*68 + "+")
    log.info("|  PIPELINE COMPLETE                                                |")
    log.info("+" + "="*68 + "+")
    log.info(f"  Wall time   : {h}h {m}m {s}s")
    log.info(f"  Output PDB  : {os.path.abspath(OUTPUT_PDB)}")
    log.info(f"  Log         : folding_pipeline.log")
    log.info(f"  Checkpoints : {os.path.abspath(CHECKPOINT_DIR)}/")


if __name__ == "__main__":
    main()
