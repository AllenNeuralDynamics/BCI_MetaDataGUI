@echo off
CALL conda activate bci_with_suite2p
ECHO %1 %2 %3
python C:\github\BCI_MetaDataGUI\UI\export_behavior.py %1 %2 %3
ECHO  
CALL conda deactivate