import numpy as np
from cobaya.yaml import yaml_load_file
from cobaya.input import update_info
from cobaya.model import Model

try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
except:
    rank = 0
    size = 1

def mprint(*args, **kwargs):
    if rank==0:
        print(*args, **kwargs)

def get_model(yaml_file, verbose=False):
    info  = yaml_load_file(yaml_file)
    if verbose:
        mprint(info)
    updated_info = update_info(info)
    model =  Model(updated_info["params"], updated_info["likelihood"],
               updated_info.get("prior"), updated_info.get("theory"),
               packages_path=info.get("packages_path"), timing=updated_info.get("timing"),
               allow_renames=False, stop_at_error=info.get("stop_at_error", False))
    return model

class CocoaModel:
    def __init__(self, configfile, likelihood):
        self.model      = get_model(configfile)
        self.likelihood = likelihood
        self.derived = np.array(list(self.model.parameterization.derived_params()))
        mprint("Derived parameters: ", self.derived)
        
    def calculate_data_vector(self, params_values, baryon_scenario=None, return_s8=False):
        likelihood   = self.model.likelihood[self.likelihood]
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(params, want_derived=False,
                                          dependency_params=depend_list, cached=False)
        if baryon_scenario is None:
            data_vector = likelihood.get_datavector(**input_params)
        else:
            data_vector = likelihood.compute_barion_datavector_masked_reduced_dim(baryon_scenario, **input_params)
        if not return_s8:
            return np.array(data_vector), None
        else:
            derived_vals = self.model.logposterior(params_values, return_derived=True).derived
            if len(derived_vals) == len(self.derived):
                derived_dict = {k:v for k,v in zip(self.derived, derived_vals)}
                return np.array(data_vector), derived_dict["sigma8"]
            else:
                print(f'Problematic derived {derived_vals} at {params_values}')
                return np.array(data_vector), np.nan

    def calculate_logpost(self, params_values):
        likelihood   = self.model.likelihood[self.likelihood]
        input_params = self.model.parameterization.to_input(params_values)
        self.model.provider.set_current_input_params(input_params)
        for (component, index), param_dep in zip(self.model._component_order.items(), 
                                                 self.model._params_of_dependencies):
            depend_list = [input_params[p] for p in param_dep]
            params = {p: input_params[p] for p in component.input_params}
            compute_success = component.check_cache_and_compute(want_derived=False,
                                         dependency_params=depend_list, cached=False, **params)
        lpost, lpriors, llikes = self.model.logposterior(params_values, return_derived=False)
        return lpost, np.sum(lpriors), np.sum(llikes)
