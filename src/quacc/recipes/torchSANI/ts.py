"""Transition state recipes for the torchSANI code."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

import numpy as np
import torch
from monty.dev import requires

from quacc import get_settings, job, strip_decorator
from quacc.runners.ase import Runner
from quacc.schemas.ase import Summarize
from quacc.utils.dicts import recursive_dict_merge

has_torchsani = bool(find_spec("torchsani"))
has_sella = bool(find_spec("sella"))

if has_torchsani:
    from torchsani.ase_interface import Calculator as TorchSANI
    from torchsani.ase_inference import ase_atoms_to_structure
    from torchsani.constants import SANI_ENERGY_KEY, SANI_FX_KEY, SANI_FY_KEY, SANI_FZ_KEY
    from torchsani.utils import compute_hessian
    from torchsani.tools.sampling.nms import get_frequencies
    from torchsani.nn import ModelInput

if has_sella:
    from sella import Sella

if TYPE_CHECKING:
    from typing import Any

    from ase.atoms import Atoms
    from numpy.typing import NDArray

    from quacc.types import OptParams, SaniTSSchema


@job
@requires(
    has_torchsani and has_sella, 
    "torchSANI and sella must be installed. Refer to the quacc documentation."
)
def ts_job(
    atoms: Atoms,
    use_custom_hessian: bool = True,
    run_freq: bool = True,
    freq_job_kwargs: dict[str, Any] | None = None,
    opt_params: OptParams | None = None,
    additional_fields: dict[str, Any] | None = None,
    **calc_kwargs,
) -> SaniTSSchema:
    """
    Perform a transition state (TS) job using the given atoms object with torchSANI and Sella.

    Parameters
    ----------
    atoms
        The atoms object representing the system.
    use_custom_hessian
        Whether to use a custom Hessian matrix.
    run_freq
        Whether to run the frequency job.
    opt_params
        Dictionary of custom kwargs for the optimization process. For a list
        of available keys, refer to [quacc.runners.ase.Runner.run_opt][].
    additional_fields
        Additional fields to add to the results dictionary.
    **calc_kwargs
        Dictionary of custom kwargs for the torchSANI calculator. Set a value to
        `quacc.Remove` to remove a pre-existing key entirely. For a list of available
        keys, refer to the `torchsani.ase_interface.Calculator` calculator.

    Returns
    -------
    SaniTSSchema
        Dictionary of results. See the type-hint for the data structure.
    """
    additional_fields = additional_fields or {}
    freq_job_kwargs = freq_job_kwargs or {}
    settings = get_settings()

    calc_defaults = {
        "model": None,  # Model should be provided via calc_kwargs
        "atom_types": {},  # Atom types should be provided via calc_kwargs
        "compute_charge_grad": True,  # Needed for proper Hessian calculation
    }
    opt_defaults = {
        "optimizer": Sella,  # Use Sella optimizer for TS search
        "optimizer_kwargs": {
            "trajectory": "sella.traj",
            "logfile": "sella.log",
            "order": 1,  # Critical parameter for TS search (first-order saddle point)
        }
    }

    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)
    opt_flags = recursive_dict_merge(opt_defaults, opt_params)

    if use_custom_hessian:
        opt_flags["optimizer_kwargs"] = opt_flags.get("optimizer_kwargs", {})
        opt_flags["optimizer_kwargs"]["hessian_function"] = _get_hessian
    
    # Initialize calculator
    calc = TorchSANI(**calc_flags)
    
    # Run the TS optimization with Sella
    dyn = Runner(atoms, calc).run_opt(**opt_flags)
    
    # Create summary
    opt_ts_summary = Summarize(additional_fields={"name": "torchSANI TS"} | additional_fields).opt(dyn)
    
    # Run frequency calculation if requested
    freq_summary = None
    if run_freq:
        try:
            # Get the final atoms and make sure the calculator is attached
            final_atoms = opt_ts_summary["atoms"]
            
            # Convert ASE atoms to Structure object
            st = ase_atoms_to_structure(final_atoms)
            
            # Calculate hessian directly using our _get_hessian function
            hessian = _get_hessian(final_atoms, **calc_flags)
            
                        
            # Calculate frequencies
            frequencies = get_frequencies(st, hessian)
            
            # Create a simplified frequency summary
            freq_summary = {
                "frequencies": frequencies,
                "n_imag": sum(1 for f in frequencies if f < 0)
            }
            
            print(f"\nNumber of imaginary frequencies: {freq_summary['n_imag']}")
            if freq_summary['n_imag'] == 1:
                print("Found exactly one imaginary frequency - this is a proper transition state!")
            elif freq_summary['n_imag'] > 1:
                print("Warning: Found multiple imaginary frequencies - this may not be a proper transition state")
            else:
                print("Warning: No imaginary frequencies found - this is not a proper transition state")
        except Exception as e:
            print(f"Error in direct frequency calculation: {str(e)}")
    
    opt_ts_summary["freq_job"] = freq_summary

    return opt_ts_summary


def _get_hessian(atoms: Atoms, **calc_kwargs) -> NDArray:
    """
    Calculate and retrieve the Hessian matrix for the given molecular configuration using torchSANI.
    
    Uses the built-in compute_hessian function from torchSANI.utils to calculate the
    Hessian matrix analytically through autograd.

    Parameters
    ----------
    atoms
        The ASE Atoms object representing the molecular configuration.
    **calc_kwargs
        Dictionary of custom kwargs for the torchSANI calculator.

    Returns
    -------
    NDArray
        The calculated Hessian matrix, reshaped into a 2D array.
    """
    settings = get_settings()
    calc_defaults = {
        "model": None,  # Model should be provided via calc_kwargs
        "atom_types": {},  # Atom types should be provided via calc_kwargs
        "compute_charge_grad": True,  # Required for Hessian calculation
    }
    calc_flags = recursive_dict_merge(calc_defaults, calc_kwargs)
    calc = TorchSANI(**calc_flags)
    
    # Get the necessary components from the calculator to compute the Hessian
    device = calc.device
    dtype = calc.dtype
    coordinates = torch.tensor(atoms.get_positions(), 
                               dtype=dtype, 
                               device=device, 
                               requires_grad=True)
    
    # Create a batch of size 1
    coordinates = coordinates.unsqueeze(0)
    
    # Get species tensor
    species = torch.tensor([calc._atom_types[s] for s in atoms.get_chemical_symbols()], 
                           dtype=torch.long, 
                           device=device).unsqueeze(0)
    
    # Create model input
    net_charge = torch.tensor(atoms.get_initial_charges().sum(), 
                              dtype=dtype, 
                              device=device).view(1, 1)
    
    model_input = ModelInput(species, coordinates, net_charge)
    
    # Get energy from model
    pbc = torch.tensor(atoms.get_pbc(), dtype=torch.bool, device=device)
    cell = None
    if pbc.any().item():
        cell = torch.tensor(atoms.get_cell(), dtype=dtype, device=device)
    
    output = calc._model(model_input, cell=cell, pbc=pbc)
    energy = output.energies[:, calc.inference_head_index]
    
    # Compute Hessian
    _, hessian = compute_hessian(coordinates, energy)
    
    # Return the reshaped Hessian matrix
    return hessian.squeeze(0).cpu().detach().numpy()