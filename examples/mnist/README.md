# Requirements:

To run the Jupyter notebook version of this tutorial:
1. Install the Poplar SDK >2.6 and source the enable.sh scripts for both PopART and Poplar as described in the [Getting Started guide](https://docs.graphcore.ai/en/latest/getting-started.html) for your IPU system
2. Create a virtual environment
3. Update `pip`: `pip3 install --upgrade pip`
4. If you cloned the `popxl.addons` repo, install its requirements in `popxl-addons` with `pip3 install -r requirements.txt` and add the repo to the python path. Otherwise, you can pip install `popxl.addons`: `pip3 install git+ssh://git@phabricator.sourcevertex.net/diffusion/POPXLADDONS/popxladdons.git`.
5. Install the requirements in `examples/mnist`: `pip3 install -r requirements.txt`
5. In the same virtual environment, install the Jupyter notebook server: `python -m pip install jupyter`
6. Launch a Jupyter Server on a specific port: `jupyter-notebook --no-browser --port <port number>`. Be sure to be in the virtual environment.
7. Connect via SSH to your remote machine, forwarding your chosen port:
`ssh -NL <port number>:localhost:<port number> <your username>@<remote machine>`

For more details about this process, or if you need troubleshooting, see our [guide on using IPUs from Jupyter notebooks](../../standard_tools/using_jupyter/README.md).

Intellisense can be very useful to understand the code, since you can see functions and class descriptions by hovering over them and easily jump
to their definitions. If you want to enable it in this tutorial, read how to configure VSCode properly: [VSCodeSetup.md](VSCodeSetup.md)
