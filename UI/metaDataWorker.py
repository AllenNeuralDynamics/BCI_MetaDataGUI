from PyQt6.QtCore import QRunnable, pyqtSignal, QObject
import traceback, os, json, traceback, shutil, requests
from pathlib import Path, PurePosixPath
import numpy as np
from datetime import datetime
from glob import glob
from aind_metadata_mapper.bergamo.session import ( BergamoEtl, 
                                                  JobSettings,
                                                  )
import bergamo_rig
from typing import Optional
from aind_data_transfer_service.models.core import (
    UploadJobConfigsV2,
    SubmitJobRequestV2,
    Task
)
from aind_data_schema_models.modalities import Modality
from aind_data_schema_models.platforms import Platform
import subprocess
from main_utility import *


class WorkerSignals(QObject):
    stepComplete = pyqtSignal(str)      
    nextStep = pyqtSignal(str)          
    allComplete = pyqtSignal()          
    transmitData = pyqtSignal(tuple)
    error = pyqtSignal(str)
    
class metaDataWorker(QRunnable):
    def __init__(self, signals, paramDict):
        super().__init__()
        self.signals = signals
        self.params = paramDict
        self.sessionData = {}
        self.mouseDict = {}
    
    def run(self):
        WRname = self.params.get('WRname')
        dateEnteredAs = self.params.get('date')
        try:
            dateTimeObj = datetime.strptime(dateEnteredAs,'%Y-%m-%d').date()
            dateFileFormat = dateTimeObj.strftime('%m%d%y') #mmddyy]
        except ValueError:
            dateFileFormat = str(dateEnteredAs) #already entered as mmddyy
        stagingDir = 'Y:/' 
        sessionFolder = Path(self.params.get('pathToRawData') + f'/{WRname}/{dateFileFormat}') #Y:/customName/mmddyy

        with open(Path(stagingDir + '/init.json'), 'r') as f:
            self.sessionData = json.load(f)
        with open(Path(stagingDir + '/mouseDict.json'), 'r') as f:
            self.mouseDict = json.load(f)
        
        self.sessionData['subject_id'] = self.params.get('subjectID')
        self.sessionData['data_streams'][0]['light_sources'][0]['wavelength'] = self.params.get('wavelength')
        self.sessionData['data_streams'][1]['ophys_fovs'][0]['imaging_depth'] = self.params.get('imagingDepth')
        self.sessionData['experimenter_full_name'][0] = self.params.get('experimenterName')
        self.sessionData['notes'] = self.params.get('notes')

        #### Logging Pipeline ####
        if '/' in WRname or ' ' in WRname:
            self.signals.error.emit('Error in typing WR Name, either a tab, enter, or space was pressed -- aborting process')
            return
        if WRname in self.mouseDict:
            self.signals.stepComplete.emit(f'{WRname} is an existing mouse')
        else:
            self.signals.stepComplete.emit(f'{WRname} is a new mouse, logging in dictionary')
            self.mouseDict[WRname] = self.params.get('subjectID')
            with open(Path(stagingDir + '/mouseDict.json'), 'w') as f:
                json.dump(self.mouseDict, f)
        
        #Step 2: check if sessionFolder exists
        stagingMouseSessionPath = f'Y:/{WRname}/{dateFileFormat}/'
        rawDataPath = f'Y:/{WRname}/{dateFileFormat}/pophys'

        if os.path.exists(sessionFolder):  
            Path(stagingMouseSessionPath).mkdir(parents=True, exist_ok=True)                                    
            behavior_folder_staging = Path.joinpath(Path(stagingMouseSessionPath),Path('behavior'))
            behavior_video_folder_staging = Path.joinpath(Path(stagingMouseSessionPath),Path('behavior_video'))
            Path(behavior_folder_staging).mkdir(parents=True, exist_ok=True)                                    
            Path(behavior_video_folder_staging).mkdir(parents=True, exist_ok=True)                              
            self.signals.stepComplete.emit('Staging Folders Created')


        else:
            if not os.path.exists(stagingMouseSessionPath):
                self.signals.error.emit('Session Folder Specified does not exist -- aborting process')
            else:
                self.signals.error.emit('Staging Directory not found -- aborting process')
            return

        #Step 4: run extract_behavior using raw data found in sessionFolder
        try:
            self.signals.nextStep.emit('Extracting Behavior')
            rc = extract_behavior(WRname, rawDataPath, behavior_folder_staging)
            behavior_fname = f"{Path(sessionFolder).name}-bpod_zaber.npy"
            self.signals.stepComplete.emit('Behavior Data Extracted Successfully')
        except Exception:
            self.signals.error.emit('Error extracting behavior -- check traceback -- aborting process')
            traceback.print_exc() 
            return

        #Step 5: Generate Rig JSON
        try:
            self.signals.nextStep.emit('Generating Rig JSON')
            rig_json = bergamo_rig.generate_rig_json()
            with open(Path(stagingMouseSessionPath).joinpath(Path('rig.json')), 'w') as json_file:
                json_file.write(rig_json)
            self.signals.stepComplete.emit('Rig JSON Created Successfully!')
        except Exception:
            self.signals.error.emit('Error generating rig json -- check traceback -- aborting process')
            traceback.print_exc() 
            return
        
        #Step 6: Generate Session JSON
        try:
            self.signals.nextStep.emit('Generating Session JSON')
            scratchInput = Path(self.params.get('pathToRawData') + f'/{WRname}/{dateFileFormat}/pophys')
            try:
                behavior_data, hittrials, goodtrials, behavior_task_name, is_side_camera_active, is_bottom_camera_active,starting_lickport_position = prepareSessionJSON(behavior_folder_staging, behavior_fname)
            except:
                print('no-learning session?')
                behavior_data, hittrials, goodtrials, behavior_task_name, is_side_camera_active, is_bottom_camera_active,starting_lickport_position = prepareSessionJSON(behavior_folder_staging, behavior_fname,nobehavior=True)# there is probably no behavior
            user_settings = JobSettings(    input_source = Path(scratchInput), #date folder local i.e. Y:/BCI93/101724/pophys
                                            output_directory = Path(stagingMouseSessionPath), #staging dir folder scratch  i.e. Y:/BCI93/101724
                                            experimenter_full_name = [str(self.sessionData['experimenter_full_name'][0])],
                                            subject_id  = str(int(self.sessionData['subject_id'])),
                                            imaging_laser_wavelength = int(self.sessionData['data_streams'][0]['light_sources'][0]['wavelength']),
                                            fov_imaging_depth = int(self.sessionData['data_streams'][1]['ophys_fovs'][0]['imaging_depth']),
                                            fov_targeted_structure = self.params.get('targetedStructure'), 
                                            notes = str(self.sessionData['notes']),
                                            session_type = "BCI",
                                            iacuc_protocol = "2109",
                                            rig_id = "442_Bergamo_2p_photostim",
                                            behavior_camera_names = np.asarray(["Side Face Camera","Bottom Face Camera"])[np.asarray([is_side_camera_active,is_bottom_camera_active])].tolist(),
                                            imaging_laser_name = "Chameleon Laser",
                                            photostim_laser_name  = "Monaco Laser",
                                            photostim_laser_wavelength =  1035,
                                            starting_lickport_position = starting_lickport_position,
                                            behavior_task_name = behavior_task_name,
                                            hit_rate_trials_0_10 = np.nanmean(hittrials[goodtrials][:10]),
                                            hit_rate_trials_20_40 = np.nanmean(hittrials[goodtrials][20:40]),
                                            total_hits  = sum(hittrials[goodtrials]),
                                            average_hit_rate = sum(hittrials[goodtrials])/sum(goodtrials),
                                            trial_num = sum(goodtrials))
            etl_job = BergamoEtl(job_settings=user_settings,)
            session_metadata = etl_job.run_job()
            self.signals.stepComplete.emit('Session JSON Created Successfully!')
        except Exception:
            self.signals.error.emit('Error generating session json -- check traceback -- aborting process')
            traceback.print_exc() 
            return
        
        #Step 7: Stage Videos
        try:
            self.signals.nextStep.emit('Staging Videos')
            stagingVideos(behavior_data, behavior_video_folder_staging)
            self.signals.stepComplete.emit('Video Staged Successfully!')
        except Exception:
            self.signals.error.emit('Error Staging Videos -- check traceback -- aborting process')
            traceback.print_exc() 
            pass
        
        #Step 8: Create and display PDFs
        try:
            self.signals.nextStep.emit('Making PDFs')
            createPDFs(stagingMouseSessionPath, behavior_data, str(self.sessionData['subject_id']), dateEnteredAs,WRname)
            self.signals.stepComplete.emit('PDFs Successfully Made!')
        except Exception:
            self.signals.error.emit('Error Generating PDFs -- check traceback -- aborting process')
            traceback.print_exc() 
            return

        #Complete!
        self.signals.transmitData.emit( (WRname, dateEnteredAs))
        self.signals.nextStep.emit(f'{WRname} is complete!')
        return



class transferToScratchWorker(QRunnable):
    def __init__(self, signals, pathDict):
        super().__init__()
        self.signals = signals
        self.params = pathDict
    
    def run(self):
        startTime = datetime.now()
        localPath = self.params.get('localPath')
        scratchPath = self.params.get('pathToRawData')
        thisMouse = self.params.get('WRname')
        dateEnteredAs = self.params.get('date')
        try:
            dateTimeObj = datetime.strptime(dateEnteredAs,'%Y-%m-%d').date()
            dateFileFormat = dateTimeObj.strftime('%m%d%y') #mmddyy
        except ValueError:
            dateFileFormat = str(dateEnteredAs) 
        sourceDir = localPath+f'/{thisMouse}/{dateFileFormat}'
        destinationDir = scratchPath+f'/{thisMouse}/{dateFileFormat}/pophys'
        self.signals.nextStep.emit('Copying Raw Data To Scratch - Transfer Worker')
        
        if not os.path.exists(destinationDir):
            shutil.copytree(sourceDir, destinationDir)
            
            
            finishTime = datetime.now()
            deltaT = (finishTime - startTime)
            self.signals.nextStep.emit('Data Successfully copied to scratch')
            self.signals.nextStep.emit(f'This took: {deltaT}')
        else:
            self.signals.nextStep.emit('Data Already Exists on Scratch')
            finishTime = datetime.now()
            deltaT = (finishTime - startTime)
            self.signals.nextStep.emit(f'This took: {deltaT}')
        self.signals.nextStep.emit('robocopy check')
        try:
            options = "/MIR /FFT /Z /R:5 /W:1 /NDL /NFL"
            robocopy_command = f"robocopy {sourceDir} {destinationDir} {options}"
            result = subprocess.run(robocopy_command, shell=True, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # Check the return code
            if result.returncode == 0:
                self.signals.nextStep.emit("No files were copied (source and destination are identical).")
            elif result.returncode == 1:
                self.signals.nextStep.emit("All files were copied successfully.")
            elif result.returncode == 2:
                self.signals.nextStep.emit("Extra files were deleted.")
            elif result.returncode == 3:
                self.signals.nextStep.emit("Some files were copied successfully, and some were skipped.")
            else:
                self.signals.nextStep.emit(f"Robocopy encountered an error. Exit code: {result.returncode}")
                print(result.stderr)
            self.signals.nextStep.emit('robocopy check done')
        except Exception as e:
            self.signals.nextStep.emit(f"An unexpected error occurred: {e}")
            self.signals.nextStep.emit('robocopy check failed')
        

#currently this is just for uploading 1 job to cloud
class cloudTransferWorker(QRunnable):
    def __init__(self, signals, params):
        super().__init__()
        self.signals = signals
        self.params = params
    def run(self):
        self.signals.nextStep.emit('Sending Data To The Cloud')
        thisMouse = self.params.get('WRname')
        dateEnteredAs = self.params.get('date')
        subject_id = self.params['subjectID']
        service_url = "http://aind-data-transfer-service/api/v2/submit_jobs" # For testing purposes, use http://aind-data-transfer-service-dev/api/v2/submit_jobs
        project_name = "Brain Computer Interface"
        s3_bucket = "default"
        platform = Platform.SINGLE_PLANE_OPHYS
        job_type = "default"
        print(self.params['sessionStart'])

        #acquisition_datetime = datetime.fromisoformat("2024-10-23T15:30:39") #find correct way to state the acq_datetime
        codeocean_pipeline_id = "a2c94161-7183-46ea-8b70-79b82bb77dc0"
        codeocean_pipeline_mount: Optional[str] = "ophys"

        pophys_task = Task(
            job_settings={
                "input_source": f"//allen/aind/scratch/BCI/2p-raw/{thisMouse}/{dateEnteredAs}/pophys"
            }
        )
        behavior_video_task = Task(
            job_settings={
                "input_source": f"//allen/aind/scratch/BCI/2p-raw/{thisMouse}/{dateEnteredAs}/behavior_video"
            }
        )
        behavior_task = Task(
            job_settings={
                "input_source": f"//allen/aind/scratch/BCI/2p-raw/{thisMouse}/{dateEnteredAs}/behavior"
            }
        )

        modality_transformation_settings = {
            Modality.POPHYS.abbreviation: pophys_task,
            Modality.BEHAVIOR_VIDEOS.abbreviation: behavior_video_task,
            Modality.BEHAVIOR.abbreviation: behavior_task
        }

        gather_preliminary_metadata = Task(
            job_settings={
                "metadata_dir": (
                    f"/allen/aind/scratch/BCI/2p-raw/{thisMouse}/{dateEnteredAs}"
                )
            }
        )

        pophys_codeocean_pipeline_settings = Task(
            skip_task=False,
            job_settings={
                "pipeline_monitor_settings": {
                    "run_params": {
                        "data_assets": [{"id": "", "mount": codeocean_pipeline_mount}],
                        "pipeline_id": codeocean_pipeline_id,
                    }
                }
            }
        )

        codeocean_pipeline_settings = {
            Modality.POPHYS.abbreviation: pophys_codeocean_pipeline_settings
        }

        #adding codeocean capsule ID and mount

        upload_job_configs = UploadJobConfigsV2(
            job_type=job_type,
            s3_bucket=s3_bucket,
            platform=platform,
            subject_id=subject_id,
            acq_datetime=datetime.strptime(
                self.params['sessionStart'], "%Y-%m-%dT%H:%M:%S.%f%z"
            ).strftime("%Y-%m-%d %H:%M:%S"),
            modalities=[Modality.POPHYS, Modality.BEHAVIOR, Modality.BEHAVIOR_VIDEOS],
            project_name=project_name,
            tasks={
                "modality_transformation_settings": modality_transformation_settings,
                "gather_preliminary_metadata": gather_preliminary_metadata,
                "codeocean_pipeline_settings": codeocean_pipeline_settings,
            },
        )

        upload_jobs = [upload_job_configs]
        submit_request = SubmitJobRequestV2(upload_jobs=upload_jobs)
        post_request_content = submit_request.model_dump(
            mode="json", exclude_none=True
        )
        
        #Submit request
        submit_job_response = requests.post(url=service_url, json=post_request_content)
        print(submit_job_response.status_code)
        print(submit_job_response.json())
        self.signals.nextStep.emit('Data Sent!')
