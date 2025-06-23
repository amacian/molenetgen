# Mole Network

Mole network is a tool that supports the process of generating Backbone, Metro core and Metro aggregation segments of a network.
It is written in Python.

This work has been developed under R&D projects:
- PID2022-136684OB-C21 funded by the Spanish Ministry of Science and Innovation MCIN/AEI/10.13039/501100011033
- EU Allegro HORIZON project (grant No.101092766) funded under HORIZON-CL4-2021-DIGITAL-EMERGING-01 Call (https://www.allegro-he.eu).

To cite this project use:
A. Sánchez-Macián, N. Koneva, M. Quagliotti, J.M. Rivas-Moscoso, F. Arpanaei,
J. A. Hernández, J.P. Fernández-Palacios 
"Generating realistic optical topologies for techno-economic studies using MoleNetwork",
Submitted to Journal of Optical Communications and Networking.

To download and execute the tool, create a directory to hold the project and run:
git clone https://github.com/amacian/molenetgen

Then, to install the dependencies, execute:
pip install -r requirements.txt
(you may prefer to create a virtual environment - venv - first and install them there)

The repository includes several files with the following being the main scripts to run the tool:

- A main.py file to run the GUI for Backbone network generation. 
- A mainMetroCore.py file for the Metro Core generator GUI.
- A mainMetroAggregation.py file for the Metro Aggregation network generator GUI

You may run any of these scripts by executing python3: e.g. "python3 main.py".
You can also use an IDE of your preference.
