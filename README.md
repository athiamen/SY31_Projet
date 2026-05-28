source /opt/ros/jazzy/setup.bash

# Aller à la racine du workspace (dossier contenant sy31_detection/)
cd ~/ros2_ws   # ou le dossier où vous avez mis le projet

# Compiler
colcon build --packages-select sy31_detection
source install/setup.bash

# Lancer le nœud de détection
ros2 launch sy31_detection detection.launch.py

PUIS

source /opt/ros/jazzy/setup.bash

# Rejouer le bag (vitesse normale)
ros2 bag play /chemin/vers/objets_0.mcap

# Ou en boucle
ros2 bag play /chemin/vers/objets_0.mcap --loop

# Ou au ralenti (0.5x) pour mieux observer
ros2 bag play /chemin/vers/objets_0.mcap --rate 0.5