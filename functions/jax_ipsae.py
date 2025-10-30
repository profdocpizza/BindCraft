#!/usr/bin/env python3
"""
jax_ipsae.py — JAX-native ipSAE implementation (auto GPU/CPU fallback)
"""

import os, numpy as _np

# Optional: prevent JAX from grabbing all VRAM (you can adjust fraction)
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.3")

try:
    import jax
    import jax.numpy as _jnp

    backend = jax.default_backend()
    print(f"[info] Using JAX backend: {backend}")

    _JAX = True
except Exception as e:
    print(f"[warn] Falling back to NumPy due to JAX error: {e}")
    _jnp = _np
    _JAX = False

NUC_SET = {"DA","DC","DT","DG","A","C","U","G"}

def _calc_d0(L, pair_type: str):
    L = float(max(L,27))
    minv = 2.0 if pair_type == "nucleic_acid" else 1.0
    return max(minv, 1.24*(L-15.0)**(1/3) - 1.8)

def _calc_d0_array(L, pair_type: str):
    L = _jnp.maximum(_jnp.asarray(L,_jnp.float32),27.0)
    minv = 2.0 if pair_type=="nucleic_acid" else 1.0
    return _jnp.maximum(minv,1.24*_jnp.power(L-15.0,1/3)-1.8)

def _pair_type(t1,t2):
    return "nucleic_acid" if (t1=="nucleic_acid" or t2=="nucleic_acid") else "protein"

def _parse_cif(cif_path: str):
    """
    Parse mmCIF and extract:
      - residues: list of dicts with coords, chain, residue name
      - chains_vec: numpy array of chain IDs
      - cb_coords: array of side-chain proxy coords (or same as CA)
      - ca_coords: array of CA (or C1*) coords
      - token_mask: boolean mask for valid residues
    """
    residues = []
    cb_residues = []
    token_mask = []
    chains = []

    with open(cif_path, "r") as fh:
        for line in fh:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            parts = line.split()
            if len(parts) < 13:
                continue
            atom_name = parts[3]
            resname = parts[5]
            chain = parts[9]
            try:
                x, y, z = float(parts[10]), float(parts[11]), float(parts[12])
            except ValueError:
                continue

            is_token = (atom_name == "CA") or ("C1" in atom_name)
            if is_token:
                residues.append({
                    "xyz": [x, y, z],
                    "resname": resname,
                    "chain": chain
                })
                chains.append(chain)
                token_mask.append(True)
            else:
                token_mask.append(False)

            if atom_name == "CB" or (resname == "GLY" and atom_name == "CA"):
                cb_residues.append({"xyz": [x, y, z], "chain": chain})

    ca_coords = _jnp.asarray([r["xyz"] for r in residues], dtype=_jnp.float32)
    cb_coords = _jnp.asarray([r["xyz"] for r in cb_residues], dtype=_jnp.float32) if cb_residues else ca_coords
    chains_vec = _np.array(chains)
    token_mask = _np.array(token_mask, dtype=bool)
    return residues, chains_vec, cb_coords, ca_coords, token_mask

def _classify_chains(chains):
    # crude: proteins unless letter N
    return {c:"protein" for c in _np.unique(chains)}

def compute_ipsae_from_files(pae_npz_path,cif_path,pae_cutoff=10.0,dist_cutoff=15.0):
    pae=_jnp.asarray(_np.load(pae_npz_path)["pae"],_jnp.float32)
    _, chains, _, _, _ = _parse_cif(cif_path)

    uniq=_np.unique(chains)
    types=_classify_chains(chains)
    N=len(chains)
    mask={ch:_jnp.asarray(chains==ch) for ch in uniq}
    pae_mask=pae<float(pae_cutoff)

    res_asym={c1:{} for c1 in uniq}
    chn_asym={c1:{} for c1 in uniq}
    dom_asym={c1:{} for c1 in uniq}

    for c1 in uniq:
        m1=mask[c1]
        for c2 in uniq:
            if c1==c2: continue
            m2=mask[c2]
            pt=_pair_type(types[c1],types[c2])
            n0chn=int(m1.sum()+m2.sum())
            d0chn=_calc_d0(n0chn,pt)
            ptm=1/(1+(pae/d0chn)**2)
            valid=m2[None,:]&pae_mask
            vals=[]
            for i in range(N):
                if not bool(m1[i]): continue
                vp=valid[i]
                if vp.any():
                    vals.append(float(ptm[i,vp].mean()))
            if not vals: vals=[0.0]
            chn_asym[c1][c2]=float(max(vals))
            res_asym[c1][c2]=chn_asym[c1][c2]
            dom_asym[c1][c2]=chn_asym[c1][c2]
    return {"ipsae_d0res_asym":res_asym,
            "ipsae_d0chn_asym":chn_asym,
            "ipsae_d0dom_asym":dom_asym,
            "meta":{"JAX_AVAILABLE":_JAX}}

if __name__=="__main__":
    import sys,json
    print(json.dumps(compute_ipsae_from_files(sys.argv[1],sys.argv[2]),indent=2))