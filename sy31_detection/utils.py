from typing import Any, Iterable

import numpy as np
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.parameter_service import Parameter, SetParametersResult
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py.point_cloud2 import create_cloud
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

Cylinder = tuple[float, float, float]
BBox = tuple[float, float, float, float]
Polyline = Iterable[tuple[float, float]]

PC2FIELDS = [
    PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    PointField(name="intensity", offset=12, datatype=PointField.FLOAT32, count=1),
    PointField(name="clusterId", offset=16, datatype=PointField.FLOAT32, count=1),
]


def declare_param(object: Node, name: str, default_value: Any) -> None:
    def callback(params: Iterable[Parameter]) -> SetParametersResult:
        for param in params:
            object.get_logger().info(f"Setting parameter '{param.name}' to {param.value}")
            setattr(object, param.name, param.value)

        return SetParametersResult(successful=True)

    if len(object._on_set_parameters_callbacks) < 2:
        object.add_on_set_parameters_callback(callback)

    object.declare_parameter(name, default_value)


def make_pointcloud2(
    header: Header, x: np.ndarray, y: np.ndarray, i: np.ndarray, c: np.ndarray = None
) -> PointCloud2:
    """Builds a PointCloud2 message from four arrays: x and y positions, intensity and cluster ID

    :param header: Input header to propagate
    :param x,y,i,c: Arrays of position, intensity and cluster ID of shape [x0, x1, ..]
    :return: 3D point cloud (x, y, z, i, c) born from the concatenation of the four arrays.
    """
    zeros = np.zeros(len(x))
    if c is None:
        c = zeros

    assert len(x) == len(y) == len(i) == len(c), (
        "Number of intensity/class does not match number of points: "
        f"({len(x)}, {len(y)}, {len(i)}, {len(c)})"
    )

    points = np.vstack((x, y, zeros, i, c)).T
    return create_cloud(header, PC2FIELDS, points)


def make_markers(
    header: Header, shapes: Iterable[Cylinder | BBox | Polyline], width: float = None
) -> MarkerArray:
    """Builds a cylinder marker for each passed cylinder.

    :param header: Input header to propagate
    :param cylinders: Array of cylinder parameters: [[x0, y0, radius0], ...]
    :param eps: Optional width for the line strip
    :return: MarkerArray message that cleans up everything and shows a cylinder per param.
    """
    markers = MarkerArray()
    markers.markers.append(Marker(header=header, action=Marker.DELETEALL))

    for c, shape in enumerate(shapes):
        rainbow = c / len(shapes)
        marker = Marker(header=header, action=Marker.ADD, id=int(c + 1))
        (col := marker.color).r, col.g, col.b, col.a = 1.0 - rainbow, rainbow, 0.0, 0.5

        if isinstance(shape, list):
            marker.type = Marker.LINE_STRIP
            marker.points.extend([Point(x=float(x), y=float(y)) for x, y in shape])
            marker.scale.x, marker.scale.y, marker.scale.z = 2.0 * width, 2.0 * width, 0.3
        elif len(shape) == 3:
            x, y, radius = shape
            marker.type = Marker.CYLINDER
            marker.pose.position.x, marker.pose.position.y = float(x), float(y)
            marker.scale.x, marker.scale.y, marker.scale.z = 2.0 * radius, 2.0 * radius, 0.3
        elif len(shape) == 4:
            x, y, width, lenght = shape
            marker.type = Marker.CUBE
            marker.pose.position.x, marker.pose.position.y = float(x), float(y)
            marker.scale.x, marker.scale.y, marker.scale.z = float(width), float(lenght), 0.3

        markers.markers.append(marker)

    return markers
