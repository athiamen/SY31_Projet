"""
detection.launch.py
--------------------
Lance le nœud de détection d'objets avec les paramètres du fichier YAML.
"""

import os
from pathlib import Path
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    pkg_dir = get_package_share_directory("sy31_detection")
    config  = os.path.join(pkg_dir, "config", "detection_params.yaml")

    detection_node = Node(
        package    = "sy31_detection",
        executable = "detection_node",
        name       = "sy31_detection_node",
        output     = "screen",
        parameters = [config],
    )

    return LaunchDescription([detection_node])
