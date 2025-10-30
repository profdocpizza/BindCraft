from .jax_ipsae import _jnp, _calc_d0_array

def _get_pae_matrix(outputs):
    """Safely extracts the PAE matrix from the model outputs."""
    if outputs is None:
        return None

    # First, try to get the 'predicted_aligned_error' dictionary
    pae_dict = outputs.get("predicted_aligned_error")

    # If pae_dict exists and is not None, try to get 'pae' from it
    if pae_dict is not None:
        return pae_dict.get("pae")

    # If that fails, fall back to getting 'pae' from the top-level outputs
    return outputs.get("pae")

def ipsae_d0res_asym_loss(inputs, outputs, align_chain="A", score_chain="B", pae_cutoff=10.0):
    """Calculates the ipSAE loss, vectorized for performance."""
    pae = _get_pae_matrix(outputs)
    if pae is None:
        return 0.0

    # Ensure inputs["asym_id"] is a JAX array for efficient processing
    chains = _jnp.asarray(inputs["asym_id"])

    # Create boolean masks for the specified chains
    mask_align = (chains == ord(align_chain.upper()))
    mask_score = (chains == ord(score_chain.upper()))

    # Calculate d0, a scaling factor based on chain lengths
    num_align = mask_align.sum()
    num_score = mask_score.sum()
    d0 = _calc_d0_array(num_align + num_score, "protein")

    # Calculate ptm, a score based on the PAE and d0
    ptm = 1.0 / (1.0 + (pae / d0) ** 2)

    # Create a 2D mask to select valid interactions
    # Valid interactions are between the align chain and the score chain, where PAE is below the cutoff
    valid_mask = (mask_align[:, None] & mask_score[None, :]) & (pae < pae_cutoff)

    # Apply the mask to the ptm matrix, zeroing out invalid entries
    masked_ptm = _jnp.where(valid_mask, ptm, 0)

    # Calculate the mean ptm score for each residue in the align chain
    # The sum is taken along the score chain axis (axis=1)
    # We add a small epsilon to the denominator to avoid division by zero
    sum_per_residue = masked_ptm.sum(axis=1)
    count_per_residue = valid_mask.sum(axis=1)
    mean_per_residue = sum_per_residue / (count_per_residue + 1e-8)

    # The final ipSAE score is the maximum of these mean scores
    ipSAE = _jnp.max(mean_per_residue)

    # Return the negative ipSAE, as the goal is to minimize this value
    return -ipSAE

def ipsae_d0chn_asym_loss(inputs, outputs, align_chain="A", score_chain="B", pae_cutoff=10.0):
    # This loss is identical to ipsae_d0res_asym_loss as per the reference implementation
    return ipsae_d0res_asym_loss(inputs, outputs, align_chain, score_chain, pae_cutoff)

def ipsae_d0dom_asym_loss(inputs, outputs, align_chain="A", score_chain="B", pae_cutoff=10.0):
    # This loss is identical to ipsae_d0res_asym_loss as per the reference implementation
    return ipsae_d0res_asym_loss(inputs, outputs, align_chain, score_chain, pae_cutoff)
