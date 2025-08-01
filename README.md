# MetaDataGUI
This GUI organizes and creates AIND metadata for 2p imaging (scanimage) + behavior (pybpod) + videography (pyspincapture) experiments on our Bergamo 2p microscope.




Data is saved on VAST as follows: <br>
```
/allen/aind/scratch/BCI/2p-raw/
                              |- mouseWRname/session_date/
                                                        |- behavior/
                                                                    |- raw & exported behavior info
                                                        |- behaviorvideo/
                                                                    |-side/
                                                                    |-bottom/ 
                                                        |- pophys/
                                                                    |- raw tiffs and other imaging files
                                                        |- rig.json 
                                                        |- session.json
```
# Installation
Go through all the steps.


## Install AIND metadata dependencies
```
conda create -n codeocean python=3.11
conda activate codeocean
pip install codeocean aind_data_schema_models aind-data-transfer-service
```

## Install Pybpod and scanimage dependencies
go somewhere you want these codes to be
```
git clone https://github.com/rozmar/suite2p.git
cd suite2p
conda env create -f environment.yml
conda activate bci_with_suite2p
pip install -e .
cd ..
git clone https://github.com/kpdaie/BCI_analysis.git
cd BCI_analysis
pip install -e .
```

## Open GUI
where you want gui to be located (and preferably in miniconda)
```
conda activate codeocean
git clone https://github.com/LKINSEY/MetaDataGUI.git
cd metaDataGUI\UI
python metaDataGUI_updateInProgress.py
```
running python in miniconda or bash can help verify that it is working.
However, a shortcut to a batch file is now included. Move this shortcut 
to the user's desktop, and double click the shortcut -- should open
the gui with just a button click after a few seconds of loading
everything.

1.) run the bat file first, this sets the bat file's working directory.
2.) copy the shortcut to desktop
3.) double click on desktop shortcut
4.) GUI should open


##AIND USAGE ONLY
map: \\allen\aind\scratch\BCI\2p-raw to Y: drive


# Usage
COMING SOON
