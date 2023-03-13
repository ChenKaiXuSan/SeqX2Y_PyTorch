# RMSim: Controlled Respiratory Motion Simulation on Static Patient Scans
A novel 3D Seq2Seq deep learning respiratory motion simulator
(RMSim) that learns from 4D-CT images and predicts future breathing phases given
a static CT image. The predicted respiratory patterns, represented by time-varying
displacement vector fields (DVFs) at different breathing phases, are modulated through
auxiliary inputs of 1D breathing traces so that a larger amplitude in the trace results in
more significant predicted deformation. Stacked 3D-ConvLSTMs are used to capture
the spatial-temporal respiration patterns. A spatial transformer deforms the static CT with the predicted DVF to
generate the predicted phase image. 10-phase 4D-CTs were
used to train and test RMSim. 

A pre-trained model in included as well as a set of 20 breathing traces and LUNA public CT dataset.
