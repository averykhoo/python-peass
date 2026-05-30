PEASS software
Version 1.0, May 12th, 2010.
By Valentin Emiya, INRIA, France.
This is a joint work by Valentin Emiya (INRIA, France), Emmanuel Vincent (INRIA, France), Niklas Harlander (University of Oldenburg, Germany), Volker Hohmann (University of Oldenburg, Germany).


*********
What for?
*********
As with BSS eval and its SDR/ISR/SIR/SAR measures, the distortion signal is decomposed into three components: target distortion, interference, artifacts; they are then used to compute several quality scores, namely OPS (Overall Perceptual Score), TPS (Target-related Perceptual Score), IPS (Interference-related Perceptual Score), APS (Artifact-related Perceptual Score).
 
This toolkit aims at providing new methods to achieve this, especially:
- a decomposition method to obtain distortion components that are closer to one can expect (e.g. not hearing the sources in the artifact component);
- quality scores that are close to what a subject would assess.

More details will be available in the coming technical report and/or under request.

*****************
In which context?
*****************
At the moment, this evaluation method is for mono/stereo audio signals. It can applied for the estimation of a source (signal at the location of the source) or of source images (signals at the location of the sensors).

***********************
Installing the software
***********************
- Unzip the file in a new directory.

- Compile some files by running compile.m under Matlab (this is optionnal and leads to faster computations).

- Extracting the distortion components can be done now. Computing quality scores requires the PEMO-Q software available at http://www.hoertech.de/web_en/produkte/downloads.shtml (not for free). See also "Using the demo version of PEMO-Q".

******************
Running an example
******************
Run example.m. The distortion components are stored as .wav files in the "example" directory.

**************
Main functions
**************
The main function is PEASS_ObjectiveMeasure.m.

Other important functions:
- "outputFilenames = extractDistortionComponents(srcFiles,estFile[,options])" decomposes the estimate file (estFile) into the sum of the true target source and of three distortion components, using the original sources (srcFiles). The target source needs to be the first element in srcFiles.
- "[qTarget, qInterf, qArtif, qGlobal] = audioQualityFeatures(decompositionFilenames)" computes the quality features using distortion components.
- "map2SubjScale" is the post-processing non-linear mapping to the subjective scale.

*********
Platforms
*********
The code can be use on any platform where Matlab is installed but PEMO-Q is running under Windows only or under Linux when 'wine' (Windows emulator) is installed.

********************************
Using the demo version of PEMO-Q
********************************
Instead of using the full version of PEMO-Q, one may use the demo version of PEMO-Q, which is freely available at: http://www.hoertech.de/web_en/produkte/downloads.shtml. It requires the user to enter a PIN code each time the function is called and only the first 4 seconds of sounds are used to assess the audio quality. As a consequence of the latter limitation, be aware that results are different from using the full PEMO-Q program if the length of sounds is greater than 4 seconds. However, this can be a good way to have approximate results.
To do so:
- download the PEMO-Q demo and unzip the archive in the current directory (the demo will be located in a new directory, e.g. 'PEMO-Q v1.1.2 demo/');
- in the 'options' input of the main function, set the fields 'options.pemoQDir' and 'options.pemoQExe' to the PEMO-Q demo directory (e.g. 'PEMO-Q v1.1.2 demo/') and to the PEMO-Q demo program name (e.g. 'audioqual_demo.exe') respectively;
- type the PIN code when asked (four times for each processed file).

*********************
Technical limitations
*********************
Please report any bug or comment to valentin.emiya@irisa.fr.
So far, some technical limitations have been noticed solved (but maybe not optimally yet):
- out of memory/large sound materials: when sounds are long, sampling frequency is high and/or sources are numerous, an "out of memory" issue may be raised. In this case, increase the "option.segmentationFactor" integer value to have the sounds segmented first, then decomposed and finally merged along the full time scale. This is due to the gammatone implementation and the current solution may be improved in the future.

**************************
How to cite this software?
**************************
When using this software, the following paper must be referred to:

Valentin Emiya, Emmanuel Vincent, Niklas Harlander and Volker Hohmann, Subjective and objective quality assessment of audio source separation, IEEE Transactions on Audio, Speech and Language Processing, submitted.

**********
References
**********
The toolbox uses an implementation of the gammatone filterbank freely available at http://www.uni-oldenburg.de/medi/download/demo/gammatone-filterbank/gammatone_filterbank-1.1.zip and related to `Frequency analysis and synthesis using a Gammatone filterbank' by V. Hohmann, Acustica/Acta Acustica, 88(3):433-442, 2002, and `Improved numerical methods for gammatone filterbank analysis and synthesis' by T. Herzke and V. Hohmann, Acustica/Acta Acustica, 93(3):498-500, 2007.
The toolbox uses the PEMO-Q software described in `PEMO-Q -- A New Method for Objective Audio Quality Assessment Using a Model of Auditory Perception' by R. Huber and B. Kollmeier, IEEE Trans. on Audio, Speech, and Language Processing, 14(6):1902-1911, 2006.

*********
Copyright
*********
The files in root directory are under:
Copyright 2010 Valentin Emiya (INRIA).

The code in the current directory is distributed under the terms of the GNU Public License version 3 (http://www.gnu.org/licenses/gpl.txt).

The files in the directory named "gammatone" are under Copyright (C) 2002 2003 2006 2007 AG Medizinische Physik, Universitaet Oldenburg, Germany, http://www.physik.uni-oldenburg.de/docs/medi. See file "gammatone/README.txt".

