# NFDE : Neural Field Differential Equations
Practical ROS2 implementation of the project "Obstacle Avoidance using Neural Fields" made in the context of the course "Brain Inspired Computing" in the University of Liège. 

---

The structure of the project is the following one : 

![](figures/structure.png)

The **NFDE** node is the efficient C++ node implementing the Neural Field Equations in a very generic way, while the other nodes are specific to the TurtleBot implementation. 

--- 

Some others features are also present : 
- **Papers**: All the papers that I used during this project.
- `bic_ws` : the working ROS2 workspace. The code is more difficult to understand but I am sure that everything works in this workspace.
- **Julia Simulations** : The notebook I used to make the simulations of the turtlebot. This folder also contains the `project_A_matrix.ipynb` file that generates the c vector hard-coded in the `NFDE.cpp` file and that generates the kernel matrix.

---

Here are some others state-of-the-art real-time obstacle avoidance algorithms I found on Internet and with which the comparison could be great : 
- Artificial Potential Field method (APF)
- Vector Field Histograme (VFH)
- Dynamic Window Approach (DWA)
- Rapidly-exploring Random Tree (RRT) 
