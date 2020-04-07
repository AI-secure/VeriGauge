# Provable Robustness Approach Comparison

This is a repo which:
- compares existing provable robustness models and verificaiton approaches.
- provides uniform and simple way to run existing neural network verification approaches.

We digest several verification approach implementations and models.

In `adaptor/` we extend the interface in `adaptor/adaptor.py` to run each verification approach in a uniform way.

In `models/` we record the model structures and model weight loaders to run trained models, with weights stored in `models_weights/` by default.
    
- Due to the large size, we will upload `models_weights/` to other places. The URL will be updated later.
- Basically it is just a collection of models from existing works, so users can also recover them by themselves.

To see how to use it, we refer the users to `main.py`.

Up to now, we support MILP, LP relaxations (Fast-Lin, CROWN, CROWN-IBP), LP-Dual, SDP, and IBP.
We will support Fast-Lip, RecurJac, Diffai series (Diffai, DeepZ, RefineZono, ...) and AI2 very soon.

In the future, we also plan to provide an uniform interface to run Reluplex easily.

For recent hot randomized smoothing series approaches, we may provide a separate tool in the future.

##### Things to do before running the tool

- Before running the tool, please create an empty folder named `tmp/` under the root path.

- Before running the tool, please substitute the constants.py definition by the paths in your environment.