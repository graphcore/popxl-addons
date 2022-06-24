# Requirements:

1. Follow the instaltion instructions in the [root README](../..)
2. Install the requirements in `examples/mnist`: `pip3 install -r requirements.txt`

In additon, to run the Jupyter notebook version of this tutorial:
1. In the same virtual environment, install the Jupyter notebook server: `python -m pip install jupyter`
2. Launch a Jupyter Server on a specific port: `jupyter-notebook --no-browser --port <port number>`. Be sure to be in the virtual environment.
3. Connect via SSH to your remote machine, forwarding your chosen port:
`ssh -NL <port number>:localhost:<port number> <your username>@<remote machine>`

For more details about this process, or if you need troubleshooting, see our [guide on using IPUs from Jupyter notebooks](../../standard_tools/using_jupyter/README.md).

Intellisense can be very useful to understand the code, since you can see functions and class descriptions by hovering over them and easily jump
to their definitions. If you want to enable it in this tutorial, read how to configure VSCode properly: [VSCodeSetup.md](VSCodeSetup.md)
