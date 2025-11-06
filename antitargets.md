# Antitargets Feature

The antitargets feature in BindCraft allows for the design of binders that are simultaneously optimized for binding to a primary target and disfavored from binding to one or more "antitargets." This is useful for designing binders with improved specificity, for example, by training against homologous proteins or preventing self-association (homodimerization).

## How it Works

The antitargets feature works by modifying the loss function used during the binder design process. When one or more antitargets are specified, the following steps are taken for each design iteration:

1.  **Primary Target Loss**: The standard loss is calculated for the binder in complex with its primary target. This loss encourages strong binding and good structural properties.

2.  **Antitarget Loss**: For each specified antitarget, a separate loss is calculated. This loss is designed to penalize binding by inverting the sign of all inter-chain loss terms (e.g., `i_pae`, `i_con`, `i_ptm`). While intra-chain losses (like `plddt` and `pae_intra`) remain positive to encourage a well-folded binder, the negative inter-chain losses guide the optimization away from forming a stable interface with the antitarget.

3.  **Combined Loss**: The total loss for the optimization step is a weighted sum of the primary target loss and all antitarget losses. The `antitarget_weight` parameter controls the relative importance of the antitarget penalty.

By minimizing this combined loss, the design algorithm finds a binder sequence that is a compromise between binding the target and not binding the antitargets.

## How to Use the Antitargets Feature

To use the antitargets feature, you need to make modifications to your `settings_target` and `settings_advanced` JSON files.

### 1. Specify Antitargets in `settings_target`

In your target settings file (e.g., `settings_target/PDL1.json`), add a new key called `"antitargets"`. This key should have a list of strings as its value. Each string can be either:

*   The keyword `"self"` to penalize binder homodimerization.
*   The path to a PDB file for an antitarget protein.

**Example `settings_target/PDL1.json`:**

```json
{
    "design_path": "/content/drive/My Drive/BindCraft/PDL1/",
    "binder_name": "PDL1",
    "starting_pdb": "/content/bindcraft/example/PDL1.pdb",
    "chains": "A",
    "target_hotspot_residues": "56",
    "lengths": [65, 150],
    "number_of_final_designs": 100,
    "antitargets": ["self", "path/to/another_protein.pdb"]
}
```

### 2. Enable Antitargets in `settings_advanced`

In your advanced settings file (e.g., `settings_advanced/default_4stage_multimer.json`), you need to enable the feature and set the weight for the antitarget loss.

*   `"use_antitargets"`: Set this to `true` to enable the feature.
*   `"antitarget_weight"`: This is a floating-point number that scales the antitarget loss. A higher value will more strongly penalize binding to the antitargets. A good starting value is `0.5`.

**Example `settings_advanced/default_4stage_multimer.json`:**

```json
{
    ...
    "use_termini_distance_loss": false,
    "weights_termini_loss": 0.1,
    "use_antitargets": true,
    "antitarget_weight": 0.5,
    "enable_mpnn": true,
    ...
}
```

With these settings, BindCraft will run the design process with the antitargets feature enabled. The output will include PDB files for the antitarget complexes and the relevant antitarget metrics in the trajectory CSV file.
