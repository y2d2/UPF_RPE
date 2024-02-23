# UPF for Relative Pose estimation between two agents for unobservable motion

This repository provides the code for an Unscented Particle Filter for estimating the relative pose between two agents equipped with one UWB device each and that share VIO measurements between them.
The UPF is capable of tracking the complex solution space when the relative pose is unobservable due to the relative motion between the agents. Video of the algorithm is available on: https://youtu.be/HPpBy0R0jvI . 
Additionally, the repository also provides  measurement and simulation data as well as the implementations of an NLS and algebraic method for comparison. 

## Paper 
The preprint of the paper accompanying this algorithm can be found in the repository. 

## Usage of the UPF: 
To use the UPF use the following code: 
Import the UPF: 
```
from Code.ParticleFilter.ConnectedAgentClass import UPFConnectedAgent
```

Initialize the UPF with: 
agent_id: the id of the agent the relative pose is being estimated.
x_odom:  represents the pose of the estimating agent in its own odometry frame.
```
upf = UPFConnectedAgent(agent_id, x_odom)
```

Set the parameters of the UKF: kappa, alfa, and beta.
```
upf.set_ukf_parameters(kappa, alpha, beta)
```

Define the initial RP distribution using r, the first measured distance between the agents,
sigma_uwb, the estimated standard deviation on the UWB distance
n_azimuth: the number of discretizations in the azimuth
n_altitude: the number of discretizations in the elevation
n_heading: the number of discretizations in the heading plane. 
```
upf.split_sphere_in_equal_areas(r, sigma_uwb, n_azimuth, n_altitude, n_heading)
```
Finally each communication step, update the pose of the estimating agent:
with x_ha: the pose of the estimating agent expressed in its odometry frame
q_ha: the covariance of the VIO from the previous communication step until the current of the estimating agent.
```
upf.ha.update(x_ha, q_ha)
```

Run the estimation algorithm using:
dx_ca: the VIO odometry as shared with the estimating agent by the estimated agent
measurement: The UWB measured distance.
q_ca: covariance of the VIO from the previous communication step until the current as communicated by the estimated agent.
time_i: is important for logging the data, but is not needed to use the algorithm.
```
upf.run_model(dx_ca, measurement, q_ca, time_i=None)
```

# Data
In the Data folder, the measurements and simulation data can be found. For the measurements the rosbags with camera images are not uploaded here, however, can be made available upon request. 
