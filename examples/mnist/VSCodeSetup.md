# VSCode setup
In this page we explain how to configure your workspace in VSCode to use IntelliSense and the visual debugger

## Intellisense 

Add the folder you are working on to your VSCode workspace, either using the `Workspace: Add folder to workspace` or directly editing your `.code-workspace ` file.

Create a `settings.json` file for your folder. If you type `Preferences: Open folder settings (JSON)` the file will be created for you, inside a `.vscode`  dir. Otherwise you can create the folder and the file directly. This file allows you to specify general settings of the folder.

PopART isn't installed as a python dependency.  Hence, you need to include the relevant paths in your `settings.json`  file, adding them to  `"python.autoComplete.extraPaths"` and `"python.analysis.extraPaths"` options.

To work with `popxl.addons`:

- If you installed it using `pip` , you need to specify the interpreter path for the folder using the Python: Select Interpreter command and selecting the interpreter of the virtual environment where you installed popxl.addons, located at `<virtual_env_path>/bin/python3`. Note: you might be tempted to specify the python.`defaultInterpreterPath` in `settings.json`, but you may incur into problems since this option it's not used once an interpreter has been selected. Read also here.

- If you cloned the directory, you need to add `<path_to_addons>/popxl-addons` to your `setting.json` file 

### Settings Template

Below is the template for your `setting.json` file. If you pip-installed `popxl.addons` in a virtual environment and you have selected the appropriate interpreter for the folder you can omit the addons path.
```
{
  "python.autoComplete.extraPaths": [
                                    "<path_to_popart>/python",
                                    "<path_to_addons>/popxl-addons" 
                                    ],
  "python.analysis.extraPath":  [
                                  "<path_to_popart>/python",
                                  "<path_to_addons>/popxl-addons" 
                                ],
}
```
### Debugging

The easiest way to use the visual debugger is creating an attach configuration and then run the debugger from the command line.

- Create a simple attach configuration in the `.vscode/launch.json` of your workspace folder 

    ```
    {
        "configurations": [
        {
            "name": "Attach_debugpy",
            "type": "python",
            "request": "attach",
            "connect": {
                "host": "localhost", 
                "port": 7000
            },
        }
        ]
    }
    ```

- Setup the environment with popart and popxl.addons

- Install requirements of your scripts.

- Install the debugpy package with `python3 -m pip install --upgrade debugpy`

- Run the script with the debugger `python3 -m debugpy --wait-for-client --listen 7000 mnist.py`. The `--wait-for-client` option prevents the script from running until you attach to the process.

- From the Run and Debug toolbar (hit on the left on the play symbol and you will see the toolbar), launch your Attach_debugpy configuration
