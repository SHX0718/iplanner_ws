conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roscore

conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch vehicle_simulator vehicle_simulator.launch world_name:=garage 
conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch vehicle_simulator vehicle_simulator.launch world_name:=forest
conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch vehicle_simulator vehicle_simulator.launch world_name:=indoor
conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch vehicle_simulator vehicle_simulator.launch world_name:=campus
conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch vehicle_simulator vehicle_simulator.launch world_name:=tunnel

conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch iplanner_node data_collector.launch

conda deactivate && cd /home/shx/Developments/iplanner_ws && source devel/setup.bash && roslaunch iplanner_node iplanner_viz.launch

