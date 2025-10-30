from .jax_ipsae import _jnp, _calc_d0_array

def ipsae_d0res_asym_loss(inputs, outputs, align_chain="A", score_chain="B", pae_cutoff=10.0):
    pae = outputs.get("pae", outputs.get("predicted_aligned_error", {}).get("pae"))
    if pae is None:
        return 0.0

    chains = inputs["asym_id"]
    mask_align = chains == align_chain
    mask_score = chains == score_chain

    pae_mask = pae < pae_cutoff
    num_align = mask_align.sum()
    num_score = mask_score.sum()

    d0 = _calc_d0_array(num_align + num_score, "protein")
    ptm = 1 / (1 + (pae / d0) ** 2)

    valid_mask = mask_score[None, :] & pae_mask

    vals = []
    for i in range(len(chains)):
        if not mask_align[i]:
            continue

        valid_pos = valid_mask[i]
        if valid_pos.any():
            vals.append(ptm[i, valid_pos].mean())

    if not vals:
        return 0.0

    ipSAE = _jnp.max(_jnp.array(vals))
    return -ipSAE

def ipsae_d0chn_asym_loss(inputs, outputs, align_chain="A", score_chain="B", pae_cutoff=10.0):
    pae = outputs.get("pae", outputs.get("predicted_aligned_error", {}).get("pae"))
    if pae is None:
        return 0.0

    chains = inputs["asym_id"]
    mask_align = chains == align_chain
    mask_score = chains == score_chain

    pae_mask = pae < pae_cutoff
    num_align = mask_align.sum()
    num_score = mask_score.sum()

    d0 = _calc_d0_array(num_align + num_score, "protein")
    ptm = 1 / (1 + (pae / d0) ** 2)

    valid_mask = mask_score[None, :] & pae_mask

    vals = []
    for i in range(len(chains)):
        if not mask_align[i]:
            continue

        valid_pos = valid_mask[i]
        if valid_pos.any():
            vals.append(ptm[i, valid_pos].mean())

    if not vals:
        return 0.0

    ipSAE = _jnp.max(_jnp.array(vals))
    return -ipSAE

def ipsae_d0dom_asym_loss(inputs, outputs, align_chain="A", score_chain="B", pae_cutoff=10.0):
    pae = outputs.get("pae", outputs.get("predicted_aligned_error", {}).get("pae"))
    if pae is None:
        return 0.0

    chains = inputs["asym_id"]
    mask_align = chains == align_chain
    mask_score = chains == score_chain

    pae_mask = pae < pae_cutoff
    num_align = mask_align.sum()
    num_score = mask_score.sum()

    d0 = _calc_d0_array(num_align + num_score, "protein")
    ptm = 1 / (1 + (pae / d0) ** 2)

    valid_mask = mask_score[None, :] & pae_mask

    vals = []
    for i in range(len(chains)):
        if not mask_align[i]:
            continue

        valid_pos = valid_mask[i]
        if valid_pos.any():
            vals.append(ptm[i, valid_pos].mean())

    if not vals:
        return 0.0

    ipSAE = _jnp.max(_jnp.array(vals))
    return -ipSAE
